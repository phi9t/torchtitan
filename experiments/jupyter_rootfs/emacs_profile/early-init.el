;;; early-init.el --- TorchTitan rootfs literate verifier profile -*- lexical-binding: t; -*-

;;; Commentary:

;; Keep batch and optional interactive startup deterministic for the
;; rootfs-local literate verifier profile.

;;; Code:

(setq package-enable-at-startup nil)
(setq inhibit-startup-screen t)
(setq inhibit-startup-message t)
(setq initial-scratch-message nil)

;;; early-init.el ends here
