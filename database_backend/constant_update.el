(defun generate-ob-n4j-keywords (kw)
    
    (json-encode-alist `((,kw . ,(org-brain-keywords kw))))
    )

(defun send-org-brain-keywords-to-python (kw)
    ;; (let* ((keywords `((,kw . ,(org-brain-keywords kw))))
	   ;; (keywords2 (push `("keyword" . ,kw) keywords))
	   ;; )
    (obvz-dbus-caller "echo"
		      (generate-ob-n4j-keywords kw)
		      )
    )


(generate-ob-n4j-keywords "status")

(send-org-brain-keywords-to-python "status")

(send-org-brain-keywords-to-python "kkk")

;; definitely should have option to send list of keyword entries
;; if i just want to send 1, just send list of 1
;; should be dict for encoding to work: then also don't need keyword

;; `send-org-brain-keywords-to-python` is now just the single case, but should be same infrastructure on python side




(defun obvz-send-to-python (dict-str-to-send)
    """general function to send stuff python, input is json-encoded dict string"""
    (when (equal obvz-connection-type "dbus")
	(obvz-dbus-caller "echo" dict-str-to-send))
    (when (equal obvz-connection-type "zmq")
	(zmq-send sock dict-str-to-send))
    )


(defun obvz-dbus-caller (method &rest args)
  "call the tomboy method METHOD with ARGS over dbus"
  (apply 'dbus-call-method 
     :session                      ; use the session (not system) bus
    "com.qtpad.dbus"                ; service name
    "/cli"                         ; path name
    "com.qtpad.dbus"               ; interface name
    method args))


(setq obvz-connection-type "dbus")

(obvz-dbus-caller "echo" "stuff")

