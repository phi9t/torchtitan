;;; init.el --- TorchTitan rootfs literate verifier profile -*- lexical-binding: t; -*-

;;; Commentary:

;; Minimal Doom-compatible Emacs profile for TorchTitan rootfs verifiers.
;;
;; The profile is intentionally smaller than the host Doom configuration. It is
;; the config surface for literate verifier artifacts: Org prose, executable
;; jupyter-python blocks, recorded results, and export-capable Org setup. Today
;; it can read packages from a mounted straight.el tree; later the same package
;; root can be populated directly by the bwrap rootfs bundle.

;;; Code:

(require 'cl-lib)
(require 'subr-x)

(defconst torchtitan-literate-profile-root
  (file-name-directory (or load-file-name user-init-file))
  "Root of the repo-owned Emacs profile.")

(defvar torchtitan-literate-package-root
  (or (getenv "TORCHTITAN_EMACS_PACKAGE_ROOT")
      "/project/emacs-packages/straight")
  "Root containing straight.el package checkouts and build directories.")

(defvar torchtitan-literate-build-pattern
  (expand-file-name "build-*/*" torchtitan-literate-package-root)
  "Glob for straight.el build directories used by the verifier profile.")

(defvar torchtitan-literate-repo-load-dirs
  '("repos/org/lisp"
    "repos/org"
    "repos/jupyter"
    "repos/emacs-zmq"
    "repos/emacs-websocket"
    "repos/emacs-web-server"
    "repos/emacs-request"
    "repos/emacs-deferred"
    "repos/dash.el"
    "repos/s.el"
    "repos/f.el"
    "repos/ht.el"
    "repos/a.el"
    "repos/uuidgen-el"
    "repos/parseedn"
    "repos/compat")
  "Fallback source package directories for the verifier profile.")

(defun torchtitan-literate--add-load-path (path &optional append)
  "Add PATH to `load-path' when it exists."
  (when (file-directory-p path)
    (add-to-list 'load-path path append)))

(defun torchtitan-literate-setup-load-path ()
  "Load Org/Jupyter packages from the hermetic package root.

Prefer Org before other packages so `org.el' and `org-loaddefs.el' come from
the same tree. Use compiled straight build directories when available, and
fall back to repository source directories for package trees that are mounted
without their build output."
  (dolist (org-dir (file-expand-wildcards
                    (expand-file-name "build-*/org" torchtitan-literate-package-root)))
    (torchtitan-literate--add-load-path org-dir))
  (dolist (d (file-expand-wildcards torchtitan-literate-build-pattern))
    (when (not (string= (file-name-nondirectory (directory-file-name d)) "org"))
      (torchtitan-literate--add-load-path d t)))
  (dolist (relative torchtitan-literate-repo-load-dirs)
    (torchtitan-literate--add-load-path
     (expand-file-name relative torchtitan-literate-package-root)
     t)))

(defun torchtitan-literate-setup ()
  "Configure Emacs for rootfs-local Org/Jupyter verifier execution."
  (torchtitan-literate-setup-load-path)
  (setq user-emacs-directory torchtitan-literate-profile-root)
  (setq custom-file (expand-file-name "custom.el" temporary-file-directory))
  (setq backup-inhibited t)
  (setq auto-save-default nil)
  (setq create-lockfiles nil)
  (setq org-confirm-babel-evaluate nil)
  (require 'org)
  (require 'ob)
  (require 'ox)
  (require 'ox-html)
  (require 'ox-latex)
  (require 'jupyter)
  (require 'jupyter-server)
  (require 'ob-jupyter)
  (org-babel-jupyter-make-language-alias "torchtitan-rootfs" "python")
  (org-babel-do-load-languages
   'org-babel-load-languages
   '((emacs-lisp . t)
     (python . t)
     (shell . t)
     (jupyter . t))))

(torchtitan-literate-setup)

(provide 'torchtitan-literate-profile)

;;; init.el ends here
