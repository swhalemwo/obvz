(require 'cl)
(require 'zmq)


(defun obvis-create-children-links (parent child)
    """basic hierarchical function"""
    (if (> (length parent) 4)
	    (when (not (equal (substring parent 0 4) "cls_"))
		(concat parent " -- " child " -- isa"))
	(concat parent " -- " child " -- isa"))
    )


(defun loop-over-upper-level-nodes (parent)
    """get nodes and hierarchical links of parent node"""
    (let ((childrenx (org-brain-children parent))
	  (res ())
	  )
	(push childrenx res)
	
	(push (mapcar (lambda (child)
			  (funcall #'obvis-create-children-links parent child))
		      childrenx) res)
	;; can also push to nodes/links to function-scoped let variables
	res
	))


(defun children-specific-depth-let (node depth)
    "retrieves children and hierarchical links of node $node to level $depth"
    (let ((nodes-upper-level (list node))
	  (all-nodes (list node))
	  (all-links ())
	  (temp-res ())
	  (all-res ())
	  )

	;; for each depth step
	(dotimes (i depth)

	    ;; temp-res: contains 2 lists (nodes/links) for each upper-level-node
	    ;; idk it's a bit ugly
	    ;; but idk how to work without temp-res
	    (push (mapcar 'loop-over-upper-level-nodes nodes-upper-level) temp-res)


	    ;; push nodes/links to local vars
	    (mapc (lambda (nodex) (push nodex all-nodes)) (flatten-list (mapcar 'cdr (car temp-res))))
	    (mapc (lambda (linkx) (push linkx all-links)) (flatten-list (mapcar 'car (car temp-res))))
	    
	    (setq nodes-upper-level (flatten-list (mapcar 'cdr (car temp-res))))
	    (setq temp-res ())

	    )
	(push all-links all-res)
	(push all-nodes all-res)

	all-res
	 
	)
    )


(defun obvz-get-friend-links (nodes)
    """get links between nodes"""
    (let ((friend-links ())
	  (friend-nodes ())
	  (node ())
	  (friends ())
	  (friend ())
	  (edge-annot)
	  (res ())
	  )

	(while nodes
	    (setq node (car nodes))
	    (setq friends (org-brain-friends node))
	    (while friends
		(setq friend (car friends))
		
		(setq edge-annot (org-brain-get-edge-annotation node friend))
		(if (equal obvz-only-use-annotated-edges t)
			(progn 
			    (if (not (equal edge-annot nil))
				    (progn
					(push (concat node " -- " friend " -- " edge-annot) friend-links)
					(push friend friend-nodes)
					)
				)
			    )
		    (progn
			(push (concat node " -- " friend " -- " edge-annot) friend-links)
			(push friend friend-nodes)
			)
		    )
		    

		(setq friends (cdr friends))

		)
	    (setq nodes (cdr nodes))
	    )


	(push friend-links res)
	(push friend-nodes res)
	;; friend-links
	res
	)
    )

	


(defun obvz-create-graph-dict (obvz-include-node-texts)
    """generate graph dict from pins"""
    (interactive)

    (let (
	  (rel-nodes org-brain-pins)
	  (total-nodes ())
	  (total-links ())
	  (node-res ())
	  (rel-node ())
	  (uniq-nodes ())
	  (uniq-nodes-hierarchy ())
	  (friend-res ())
	  (class-nodes ())
	  (friend-links ())
	  (friend-nodes ())
	  (all-links ())
	  (link-string ())
	  (node-texts ())
	  (node-text-alist ())
	  (graph-dict ())
	  (current-node ())
	  )
	;; get hierarchical relations

	
	(while rel-nodes
	    
	    (setq rel-node (car rel-nodes))
	    ;; (print rel-node)

	    (setq node-res (children-specific-depth-let rel-node 8))
	    (push (car node-res) total-nodes)
	    (push (cdr node-res) total-links)

	    (setq rel-nodes (cdr rel-nodes))
	    
	    )

	;; (print total-nodes)
	(setq uniq-nodes (remove-duplicates (flatten-list total-nodes)))
	;; (print uniq-nodes)

	


	;; handle links
	(setq friend-res (obvz-get-friend-links (append uniq-nodes class-nodes)))
	(setq friend-links (cdr friend-res))
	(push friend-links total-links)
	(setq all-links (flatten-list total-links))
	(setq link-string (mapconcat 'identity all-links ";"))
	;; (message "all links there")

	;; handle nodes
	(setq friend-nodes (car friend-res))
	(setq uniq-nodes (remove-duplicates (flatten-list (list uniq-nodes friend-nodes))))
	
	;; delete cls_nodes from being there if alone
	;; not clear if effective: what if referred to as friend? 
	(setq uniq-nodes (cl-delete-if (lambda (k) (string-match-p "cls_" k)) uniq-nodes))
	
	;; (message "all nodes there")
	
	;; (setq node-string (mapconcat 'identity uniq-nodes ";"))
	;; include (or not) node texts
	(if (equal obvz-include-node-texts t)
		(progn
		    (setq node-texts (mapcar 'org-brain-text uniq-nodes))
		    (setq node-text-alist (mapcar* #'cons uniq-nodes node-texts))
		    )
	    (setq node-text-alist (mapcar* #'cons uniq-nodes (make-list (len uniq-nodes) "")))
	    )
	;; (message "node texts there")

	;; get current node
	(if (equal obvz-highlight-current-node t)
		(setq current-node (org-brain-entry-at-pt))
	    (setq current-node nil))
	    
		

	(setq graph-dict
	      `(("links" . ,all-links)
		("nodes" . ,uniq-nodes)
		("cur_node" . ,current-node)
		("node_texts" . ,node-text-alist)
		("draw_arrow_toggle" . ,obvz-draw-arrow)
		)
	      )
	graph-dict
	)
    )



;; data structure: dict with key node, value text
;; is itself dict in graph_dict


(defun obvz-switch-node-text-inclusion()
    (interactive)
    (if (equal obvz-include-node-texts t)
	    (setq obvz-include-node-texts nil)
	(setq obvz-include-node-texts t)
	)
    (obvz-update-graph)
    )



;; (define-key org-brain-visualize-mode-map "G" 'obr-viz)


(defun obvz-reposition-nodes()
    "redraw layout, either soft (apply forces to current layout) or hard (from random starting positions)"
    (interactive)
    (let ((called-prefix current-prefix-arg))
	(if (equal called-prefix nil)
		    (setq obvz-redraw-alist '(("redraw" . "soft")))
		(setq obvz-redraw-alist '(("redraw" . "hard")))
		)
	;; (zmq-send sock (json-encode-alist obvz-redraw-alist))
	(obvz-send-to-python (json-encode-alist obvz-redraw-alist))
	)
    )



(defun obvz-update-graph ()
    (interactive)

    (setq obvz-current-config (obvz-create-graph-dict obvz-include-node-texts))
    (if (not (equal obvz-current-config obvz-most-recent-config))
	    (progn
		(setq obvz-most-recent-config obvz-current-config)
		;; (zmq-send sock (json-encode-alist obvz-current-config))
		(obvz-send-to-python (json-encode-alist obvz-current-config))
		)
    )
    )


(defun obvz-start ()
    (interactive)
    (shell-command (concat "cd "obvz-dir " && python3 obr_viz_server.py " obvz-connection-type " & "))
    (setq obvz-most-recent-config ())
    )



;; connection functions

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
	
	
	    

(add-hook 'org-brain-after-visualize-hook 'obvz-update-graph) ;; automatic redrawing with org-brain


(setq obvz-connection-type "dbus")
(setq obvz-connection-type "zmq")
(setq obvz-dir "~/Dropbox/personal_stuff/obr-viz/")
(setq obvz-include-node-texts t)
(setq obvz-only-use-annotated-edges t)
(setq obvz-most-recent-config ())
(setq obvz-draw-arrow t)
(setq obvz-highlight-current-node t)


(define-key org-brain-visualize-mode-map "N" 'obvz-switch-node-text-inclusion)
(define-key org-brain-visualize-mode-map "R" 'obvz-reposition-nodes)
(define-key org-brain-visualize-mode-map "U" 'obvz-update-graph)


(setq sock (zmq-socket (zmq-context) zmq-PUB))
(zmq-bind sock "tcp://127.0.0.1:5556")

(zmq-send sock "LOL")

    
;; * dbus tests



(setq ECHO_BUS_NAME "com.qtpad.dbus")
(setq ECHO_OBJECT_PATH "/cli")
(setq ECHO_INTERFACE "com.qtpad.dbus")



(require 'dbus)
(defun djcb-call-tomboy (method &rest args)
  "call the tomboy method METHOD with ARGS over dbus"
  (apply 'dbus-call-method 
    :session                            ; use the session (not system) bus
    ECHO_BUS_NAME       ;; "org.gnome.Tomboy"      ; service name
    ECHO_OBJECT_PATH       ;; "/org/gnome/Tomboy/RemoteControl"   ; path name
    ECHO_INTERFACE ;; "org.gnome.Tomboy.RemoteControl"    ; interface name
    method args))

(djcb-call-tomboy "echo" "jjj")

(djcb-call-tomboy "echo" (json-encode-alist obvz-current-config))


