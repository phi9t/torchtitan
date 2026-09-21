;;; verify_org_babel.el --- Execute rootfs Jupyter Org Babel probes -*- lexical-binding: t; -*-
;; Copyright (c) Meta Platforms, Inc. and affiliates.
;; All rights reserved.

;;; Commentary:

;; Batch verifier for Org Babel literate probes that use jupyter-python blocks.
;; Emacs runs inside the repo-local bwrap rootfs with the repo-owned literate
;; profile loaded. Source blocks attach to the rootfs-hosted Jupyter server and
;; execute in the torchtitan-rootfs kernel.

;;; Code:

(require 'cl-lib)
(require 'subr-x)

(unless (featurep 'torchtitan-literate-profile)
  (error "torchtitan-literate-profile was not loaded; run with --init-directory"))
(require 'org)
(require 'ob)
(require 'jupyter)
(require 'jupyter-server)
(require 'ob-jupyter)

(defvar jupyter-org-verify--failures 0
  "Count of failed assertions during the current verifier run.")

(defvar jupyter-org-verify--session nil
  "Jupyter Org session string used for verifier block execution.")

(defvar jupyter-org-verify--kernel nil
  "Jupyter kernel name used for verifier block execution.")

(defvar jupyter-org-verify--retries 1
  "Number of retries for transient Org Jupyter block execution failures.")

(defvar jupyter-org-verify--client nil
  "Single Jupyter client shared by every block in the current Org file.")

(defun jupyter-org-verify--getenv (name default)
  "Return env var NAME, or DEFAULT when unset or empty."
  (let ((value (getenv name)))
    (if (and value (not (string-empty-p value))) value default)))

(defun jupyter-org-verify--check (label ok detail)
  "Print a PASS/FAIL line for LABEL according to OK with DETAIL."
  (if ok
      (princ (format "PASS %-18s %s\n" label detail))
    (setq jupyter-org-verify--failures (1+ jupyter-org-verify--failures))
    (princ (format "FAIL %-18s %s\n" label detail))))

(defun jupyter-org-verify--start-client (server kernel)
  "Start one emacs-jupyter client for KERNEL on SERVER."
  (jupyter-client
   (jupyter-kernel :server server :spec kernel)
   'jupyter-kernel-client))

(defun jupyter-org-verify--python-string (value)
  "Decode a simple Python repr string VALUE into an Emacs string."
  (let ((text (string-trim (or value ""))))
    (when (and (>= (length text) 2)
               (member (substring text 0 1) '("\"" "'"))
               (string= (substring text -1) (substring text 0 1)))
      (setq text (substring text 1 -1)))
    (setq text (replace-regexp-in-string "\\\\n" "\n" text t t))
    (setq text (replace-regexp-in-string "\\\\t" "\t" text t t))
    (setq text (replace-regexp-in-string "\\\\\"" "\"" text t t))
    (setq text (replace-regexp-in-string "\\\\'" "'" text t t))
    (replace-regexp-in-string "\\\\\\\\" "\\" text t t)))

(defun jupyter-org-verify--eval (code &optional timeout)
  "Evaluate CODE in `jupyter-org-verify--client' and return captured stdout."
  (let* ((payload (json-encode-string code))
         (wrapped
          (concat
           "import contextlib, io, traceback\n"
           "__torchtitan_org_stdout = io.StringIO()\n"
           "try:\n"
           "    with contextlib.redirect_stdout(__torchtitan_org_stdout):\n"
           "        exec(" payload ", globals(), globals())\n"
           "except Exception:\n"
           "    with contextlib.redirect_stdout(__torchtitan_org_stdout):\n"
           "        traceback.print_exc()\n"
           "    raise\n"
           "__torchtitan_org_stdout.getvalue()\n")))
    (jupyter-org-verify--python-string
     (jupyter-run-with-client jupyter-org-verify--client
       (jupyter-mlet*
           ((res (jupyter-result
                  (jupyter-message-subscribed
                   (jupyter-execute-request
                    :code wrapped
                    :store-history nil
                    :handlers nil)
                   `(("execute_reply"
                      ,(jupyter-message-lambda (status evalue)
                         (unless (equal status "ok")
                           (error "%s" (ansi-color-apply evalue)))))))
                  timeout)))
         (jupyter-return (jupyter-message-data res :text/plain)))))))

(defun jupyter-org-verify--block-name ()
  "Return the Org Babel source block name at point, or nil."
  (save-excursion
    (let ((case-fold-search t)
          (limit (save-excursion
                   (or (re-search-backward org-outline-regexp-bol nil t)
                       (point-min)))))
      (when (re-search-backward "^[ \t]*#\\+name:[ \t]*\\(.+\\)$" limit t)
        (string-trim (match-string-no-properties 1))))))

(defun jupyter-org-verify--insert-result (name result)
  "Replace the current source block result with RESULT named NAME."
  (org-babel-remove-result)
  (goto-char (org-babel-where-is-src-block-head))
  (unless (re-search-forward "^[ \t]*#\\+end_src" nil t)
    (error "cannot find end_src for %s" name))
  (end-of-line)
  (insert "\n\n#+RESULTS")
  (when name
    (insert ": " name))
  (insert "\n")
  (dolist (line (split-string (string-remove-suffix "\n" result) "\n"))
    (insert ": " line "\n")))

(defun jupyter-org-verify--execute-jupyter-blocks ()
  "Execute every jupyter-python source block in the current Org buffer."
  (let ((count 0))
    (org-babel-map-src-blocks nil
      (when (string= lang "jupyter-python")
        (setq count (1+ count))
        (goto-char (org-babel-where-is-src-block-head))
        (let ((name (or (jupyter-org-verify--block-name)
                        (format "block-%d" count))))
          (let ((attempt 0)
                (done nil)
                last-error)
            (while (and (not done) (<= attempt jupyter-org-verify--retries))
              (condition-case err
                  (progn
                    (let* ((info (org-babel-get-src-block-info))
                           (body (nth 1 info))
                           (result (jupyter-org-verify--eval body jupyter-long-timeout)))
                      (jupyter-org-verify--insert-result name result))
                    (setq done t))
                (error
                 (setq last-error err)
                 (setq attempt (1+ attempt))
                 (when (<= attempt jupyter-org-verify--retries)
                   (sleep-for 1)))))
            (if done
                (jupyter-org-verify--check "execute-block" t name)
              (jupyter-org-verify--check
               "execute-block" nil (format "%s: %S" name last-error)))))))
    (jupyter-org-verify--check "block-count" (> count 0) (number-to-string count))
    count))

(defun jupyter-org-verify--result-after-block ()
  "Return the Org result string following the source block at point."
  (when-let* ((pos (org-babel-where-is-src-block-result)))
    (save-excursion
      (goto-char pos)
      (forward-line 1)
      (let ((beg (point))
            (end (org-babel-result-end)))
        (string-trim (buffer-substring-no-properties beg end))))))

(defun jupyter-org-verify--collect-results ()
  "Return an alist mapping source block names to their result strings."
  (let (results)
    (org-babel-map-src-blocks nil
      (when (string= lang "jupyter-python")
        (goto-char (org-babel-where-is-src-block-head))
        (when-let* ((name (jupyter-org-verify--block-name))
                    (result (jupyter-org-verify--result-after-block)))
          (push (cons name result) results))))
    results))

(defun jupyter-org-verify--result-contains-p (results name needle)
  "Return non-nil when named result NAME in RESULTS contains NEEDLE."
  (when-let* ((value (cdr (assoc name results))))
    (string-match-p (regexp-quote needle) value)))

(defun jupyter-org-verify--split-checks (spec)
  "Parse SPEC as NAME=NEEDLE checks separated by semicolons."
  (cl-loop
   for item in (split-string (or spec "") ";" t "[ \t\n]+")
   when (string-match "\\`\\([^=]+\\)=\\(.*\\)\\'" item)
   collect (cons (string-trim (match-string 1 item))
                 (string-trim (match-string 2 item)))))

(let* ((org-file (jupyter-org-verify--getenv "JUPYTER_VERIFY_ORG_FILE" ""))
       (url (jupyter-org-verify--getenv "JUPYTER_VERIFY_URL" "http://127.0.0.1:8899"))
       (token (getenv "JUPYTER_VERIFY_TOKEN"))
       (kernel (jupyter-org-verify--getenv "JUPYTER_VERIFY_KERNEL" "torchtitan-rootfs"))
       (session-name (jupyter-org-verify--getenv "JUPYTER_VERIFY_ORG_SESSION" "torchtitan-literate"))
       (retries (string-to-number
                 (jupyter-org-verify--getenv "JUPYTER_VERIFY_ORG_RETRIES" "1")))
       (required-checks
        (jupyter-org-verify--split-checks
         (jupyter-org-verify--getenv
          "JUPYTER_VERIFY_REQUIRED_RESULTS"
          "rootfs-context=\"in_rootfs\": true;rootfs-context=\"shell\": \"/bin/bash\";hack-probe=\"fsdp_module\": true")))
       (session session-name))
  (setq jupyter-org-verify--session session)
  (setq jupyter-org-verify--kernel kernel)
  (setq jupyter-org-verify--retries retries)
  (unless (and token (not (string-empty-p token)))
    (princ "FAIL token              JUPYTER_VERIFY_TOKEN is empty\n")
    (kill-emacs 3))
  (unless (and org-file (file-readable-p org-file))
    (princ (format "FAIL org-file           unreadable: %s\n" org-file))
    (kill-emacs 3))
  (let* ((server (jupyter-server
                  :url url
                  :auth `(("Authorization" . ,(concat "token " token)))))
         (jupyter-api-authentication-method
          `(("Authorization" . ,(concat "token " token))))
         (org-confirm-babel-evaluate nil)
         (org-babel-default-header-args:jupyter-python
          `((:session . ,session)
            (:kernel . ,kernel)
            (:results . "replace output")
            (:exports . "both")))
         (verify-timeout
          (string-to-number
           (jupyter-org-verify--getenv "JUPYTER_VERIFY_ORG_BLOCK_TIMEOUT" "300")))
         (jupyter-default-timeout verify-timeout)
         (jupyter-long-timeout verify-timeout))
    (setq jupyter-current-server server)
    (setq jupyter-org-verify--client
          (jupyter-org-verify--start-client server kernel))
    (with-current-buffer (find-file-noselect org-file)
      (org-mode)
      (let* ((executed (jupyter-org-verify--execute-jupyter-blocks))
             (results (if (zerop jupyter-org-verify--failures)
                          (jupyter-org-verify--collect-results)
                        nil)))
        (dolist (check required-checks)
          (let ((name (car check))
                (needle (cdr check)))
            (jupyter-org-verify--check
             name
             (and (> executed 0)
                  (jupyter-org-verify--result-contains-p results name needle))
             (or (cdr (assoc name results)) "missing")))))
      (save-buffer)))
  (ignore-errors (jupyter-shutdown-kernel jupyter-org-verify--client))
  (if (zerop jupyter-org-verify--failures)
      (progn (princ "VERIFY-OK org babel assertions passed\n") (kill-emacs 0))
    (princ (format "VERIFY-FAIL %d assertion(s) failed\n" jupyter-org-verify--failures))
    (kill-emacs 1)))

;;; verify_org_babel.el ends here
