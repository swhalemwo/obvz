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
