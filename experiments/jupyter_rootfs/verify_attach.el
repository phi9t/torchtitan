;;; verify_attach.el --- Headless emacs-jupyter self-check -*- lexical-binding: t; -*-
;; Copyright (c) Meta Platforms, Inc. and affiliates.
;; All rights reserved.
;;
;; Drives emacs-jupyter against a running rootfs-hosted Jupyter server. It
;; launches the pinned kernelspec through the server API, connects an
;; emacs-jupyter client, then asserts the kernel really runs inside the bwrap
;; rootfs on GPU:
;;
;;   - sys.prefix ends with .venv-rootfs
;;   - torch.cuda.is_available() is True
;;   - torch.cuda.device_count() meets JUPYTER_VERIFY_MIN_GPUS
;;   - /opt/cuda-synth exists (rootfs CUDA marker)
;;   - CUDA_HOME == /opt/cuda-synth
;;   - HOME == /project/home (rootfs project home)
;;   - a real CUDA matmul runs
;;
;; Any failed assertion calls `kill-emacs' with a non-zero code, so verify.sh
;; propagates drift as a non-zero exit. Config comes from the environment:
;;   JUPYTER_VERIFY_URL, JUPYTER_VERIFY_TOKEN, JUPYTER_VERIFY_KERNEL,
;;   JUPYTER_VERIFY_MIN_GPUS.

(unless (featurep 'torchtitan-literate-profile)
  (error "torchtitan-literate-profile was not loaded; run with --init-directory"))
(require 'jupyter)
(require 'jupyter-server)

(defvar jupyter-verify--failures 0
  "Count of failed assertions during this batch run.")

(defvar jupyter-verify--retries 1
  "Number of retries for transient Jupyter REPL startup failures.")

(defun jupyter-verify--getenv (name default)
  "Return env var NAME or DEFAULT when unset or empty."
  (let ((v (getenv name)))
    (if (and v (not (string-empty-p v))) v default)))

(defun jupyter-verify--check (label ok detail)
  "Record assertion LABEL: pass when OK, printing DETAIL either way."
  (if ok
      (princ (format "PASS %-18s %s\n" label detail))
    (setq jupyter-verify--failures (1+ jupyter-verify--failures))
    (princ (format "FAIL %-18s %s\n" label detail))))

(defun jupyter-verify--eval (code &optional timeout)
  "Evaluate CODE in the current Jupyter client, waiting up to TIMEOUT seconds."
  (jupyter-run-with-client jupyter-current-client
    (jupyter-mlet*
        ((res (jupyter-result
               (jupyter-message-subscribed
                (jupyter-execute-request
                 :code code
                 :store-history nil
                 :handlers nil)
                `(("execute_reply"
                   ,(jupyter-message-lambda (status evalue)
                      (unless (equal status "ok")
                        (error "%s" (ansi-color-apply evalue)))))))
               timeout)))
      (jupyter-return (jupyter-message-data res :text/plain)))))

(defun jupyter-verify--start-client (server kernel)
  "Start an emacs-jupyter client for KERNEL on SERVER with bounded retry."
  (let ((attempt 0)
        client
        last-error)
    (while (and (not client) (<= attempt jupyter-verify--retries))
      (condition-case err
          (setq client
                (jupyter-client
                 (jupyter-kernel :server server :spec kernel)
                 'jupyter-kernel-client))
        (error
         (setq last-error err)
         (setq attempt (1+ attempt))
         (when (<= attempt jupyter-verify--retries)
           (sleep-for 1)))))
    (or client (signal (car last-error) (cdr last-error)))))

(let* ((url (jupyter-verify--getenv "JUPYTER_VERIFY_URL" "http://127.0.0.1:8899"))
       (token (getenv "JUPYTER_VERIFY_TOKEN"))
       (kernel (jupyter-verify--getenv "JUPYTER_VERIFY_KERNEL" "torchtitan-rootfs"))
       (verify-timeout (string-to-number
                        (jupyter-verify--getenv "JUPYTER_VERIFY_TIMEOUT" "300")))
       (retries (string-to-number
                 (jupyter-verify--getenv "JUPYTER_VERIFY_RETRIES" "1")))
       (min-gpus (string-to-number
                  (jupyter-verify--getenv "JUPYTER_VERIFY_MIN_GPUS" "1"))))
  (setq jupyter-default-timeout verify-timeout)
  (setq jupyter-long-timeout verify-timeout)
  (setq jupyter-verify--retries retries)
  (unless (and token (not (string-empty-p token)))
    (princ "FAIL token              JUPYTER_VERIFY_TOKEN is empty\n")
    (kill-emacs 3))
  (condition-case err
      (let* ((server (jupyter-server
                      :url url
                      :auth `(("Authorization" . ,(concat "token " token)))))
             (client nil))
        (setq jupyter-current-server server)
        ;; A wrong URL/token surfaces here as an error, which the handler below
        ;; turns into a non-zero exit.
        (let ((names (mapcar #'jupyter-kernelspec-name
                             (jupyter-kernelspecs server))))
          (jupyter-verify--check
           "kernelspec" (member kernel names)
           (format "%s in %S" kernel names)))
        ;; Start the pinned kernel on the server and get an emacs-jupyter
        ;; client without bootstrapping a UI REPL buffer in batch mode.
        (setq client (jupyter-verify--start-client server kernel))
        (setq jupyter-current-client client)
        (let* ((extra-eval (getenv "JUPYTER_VERIFY_EXTRA_EVAL"))
               (extra-timeout (string-to-number
                               (jupyter-verify--getenv "JUPYTER_VERIFY_EXTRA_TIMEOUT" "180")))
               (prefix (jupyter-verify--eval "import sys; sys.prefix"))
               (venv-match (jupyter-verify--eval "sys.prefix.endswith('.venv-rootfs')"))
               (cuda-avail (jupyter-verify--eval "__import__('torch').cuda.is_available()"))
               (dev-count (jupyter-verify--eval "__import__('torch').cuda.device_count()"))
               (dev-name (jupyter-verify--eval "__import__('torch').cuda.get_device_name(0)"))
               (cuda-synth (jupyter-verify--eval "__import__('os').path.exists('/opt/cuda-synth')"))
               (cuda-home (jupyter-verify--eval "__import__('os').environ.get('CUDA_HOME')"))
               (home (jupyter-verify--eval "__import__('os').environ.get('HOME')"))
               (shell (jupyter-verify--eval "__import__('os').environ.get('SHELL')"))
               (shell-pwd (jupyter-verify--eval
                           "get_ipython().getoutput('pwd')[0]"))
               (matmul (jupyter-verify--eval
                        (concat "float((__import__('torch').randn(1024,1024,device='cuda')"
                                "@__import__('torch').randn(1024,1024,device='cuda')).sum())")))
               (extra-result (when (and extra-eval (not (string-empty-p extra-eval)))
                               (jupyter-verify--eval extra-eval extra-timeout)))
               (ndev (string-to-number (or dev-count "0"))))
          (jupyter-verify--check "sys.prefix" (equal venv-match "True") prefix)
          (jupyter-verify--check "cuda.available" (equal cuda-avail "True") cuda-avail)
          (jupyter-verify--check "device_count" (>= ndev min-gpus)
                                 (format "%s (min %d)" dev-count min-gpus))
          (jupyter-verify--check "device0" (and dev-name (not (string-empty-p dev-name)))
                                 dev-name)
          (jupyter-verify--check "cuda-synth" (equal cuda-synth "True") cuda-synth)
          (jupyter-verify--check "CUDA_HOME" (equal cuda-home "'/opt/cuda-synth'")
                                 cuda-home)
          (jupyter-verify--check "HOME" (equal home "'/project/home'") home)
          (jupyter-verify--check "SHELL" (equal shell "'/bin/bash'") shell)
          (jupyter-verify--check "bang-pwd"
                                 (equal shell-pwd "'/workspace/torchtitan'")
                                 shell-pwd)
          (jupyter-verify--check "cuda-matmul"
                                 (and matmul (not (string-empty-p matmul))
                                      (not (string-prefix-p "ERR" matmul)))
                                 matmul)
          (when extra-eval
            (jupyter-verify--check "extra-eval"
                                   (and extra-result
                                        (not (string-empty-p extra-result))
                                        (not (string-prefix-p "ERR" extra-result)))
                                   extra-result)))
        (ignore-errors (jupyter-shutdown-kernel client)))
    (error
     (princ (format "FAIL attach             %S\n" err))
     (setq jupyter-verify--failures (1+ jupyter-verify--failures))))
  (if (zerop jupyter-verify--failures)
      (progn (princ "VERIFY-OK all assertions passed\n") (kill-emacs 0))
    (princ (format "VERIFY-FAIL %d assertion(s) failed\n" jupyter-verify--failures))
    (kill-emacs 1)))

;;; verify_attach.el ends here
