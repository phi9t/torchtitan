;;; verify_attach.el --- Headless emacs-jupyter self-check -*- lexical-binding: t; -*-
;; Copyright (c) Meta Platforms, Inc. and affiliates.
;; All rights reserved.
;;
;; Drives the exact emacs-jupyter `jupyter-run-server-repl' codepath that the
;; interactive `M-x jupyter-run-server-repl' uses, against a running
;; rootfs-hosted Jupyter server. It launches the pinned kernelspec, then
;; asserts the kernel really runs inside the bwrap rootfs on GPU:
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

;; Load every straight build dir onto load-path, matching how Doom assembles
;; it, then require the jupyter surface. This is the bootstrap proven to work
;; under `emacs --batch' in this environment.
(dolist (d (file-expand-wildcards "~/.emacs.d/.local/straight/build-*/*"))
  (when (file-directory-p d)
    (add-to-list 'load-path d)))
(require 'jupyter)
(require 'jupyter-server)

(defvar jupyter-verify--failures 0
  "Count of failed assertions during this batch run.")

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

(let* ((url (jupyter-verify--getenv "JUPYTER_VERIFY_URL" "http://127.0.0.1:8899"))
       (token (getenv "JUPYTER_VERIFY_TOKEN"))
       (kernel (jupyter-verify--getenv "JUPYTER_VERIFY_KERNEL" "torchtitan-rootfs"))
       (min-gpus (string-to-number
                  (jupyter-verify--getenv "JUPYTER_VERIFY_MIN_GPUS" "1"))))
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
        ;; Exact interactive codepath: start the pinned kernel on the server
        ;; and get a REPL client.
        (setq client (jupyter-run-server-repl server kernel "verify" nil nil nil))
        (setq jupyter-current-client client)
        (let* ((prefix (jupyter-eval "import sys; sys.prefix"))
               (venv-match (jupyter-eval "sys.prefix.endswith('.venv-rootfs')"))
               (cuda-avail (jupyter-eval "__import__('torch').cuda.is_available()"))
               (dev-count (jupyter-eval "__import__('torch').cuda.device_count()"))
               (dev-name (jupyter-eval "__import__('torch').cuda.get_device_name(0)"))
               (cuda-synth (jupyter-eval "__import__('os').path.exists('/opt/cuda-synth')"))
               (cuda-home (jupyter-eval "__import__('os').environ.get('CUDA_HOME')"))
               (home (jupyter-eval "__import__('os').environ.get('HOME')"))
               (matmul (jupyter-eval
                        (concat "float((__import__('torch').randn(1024,1024,device='cuda')"
                                "@__import__('torch').randn(1024,1024,device='cuda')).sum())")))
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
          (jupyter-verify--check "cuda-matmul"
                                 (and matmul (not (string-empty-p matmul))
                                      (not (string-prefix-p "ERR" matmul)))
                                 matmul))
        (ignore-errors (jupyter-shutdown-kernel client)))
    (error
     (princ (format "FAIL attach             %S\n" err))
     (setq jupyter-verify--failures (1+ jupyter-verify--failures))))
  (if (zerop jupyter-verify--failures)
      (progn (princ "VERIFY-OK all assertions passed\n") (kill-emacs 0))
    (princ (format "VERIFY-FAIL %d assertion(s) failed\n" jupyter-verify--failures))
    (kill-emacs 1)))

;;; verify_attach.el ends here
