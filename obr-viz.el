(require 'cl)



(defun obvz-flatten-list (mylist)
    "flatten list function because somehow only there in emacs 27?"
  (cond
   ((null mylist) nil)
   ((atom mylist) (list mylist))
   (t
    (append (obvz-flatten-list (car mylist)) (obvz-flatten-list (cdr mylist))))))



(defun obvz-create-children-links (parent child)
    """basic hierarchical function"""
    (concat (obvz-get-node-name parent) " -- " (obvz-get-node-name child) " -- isa"))



(defun obvz-is-node-headline (node)
    "check whether node is headline entry"
    (let ((node-state nil))
	(if (equal (type-of node) 'cons)
		(setq node-state t)
	    (setq node-state nil))
	node-state))
	
	

(defun obvz-get-node-name (node)
    "get actual name of node (not identical when headline entry)"
    (let ((node-name ""))
	(if (obvz-is-node-headline node)
		(setq node-name (nth 1 node))
	    (setq node-name node))
	node-name))
	    


(defun obvz-loop-over-upper-level-nodes (parent)
    """get nodes and hierarchical links of parent node, now also for headline entries"""
    (let ((childrenx (org-brain-children parent))
	  (node-names)
	  (res ()))
	  
	(push childrenx res)

	;; (setq childrenx2 (mapcar 'obvz-get-node-name childrenx))
	
	(cl-delete-if (lambda (k) (obvz-is-node-cls-node k)) childrenx)


	;; prevents hierarchical links from nodes to cls_ nodes
	;; also need other way: prevent cls_nodes as parents of other nodes
	;; if class node: push empty list to res
	(if (obvz-is-node-cls-node parent)
		(push () res)
	    (push (mapcar (lambda (child) (funcall #'obvz-create-children-links parent child)) childrenx) res))

	res))



	    
	
(defun obvz-is-node-cls-node (node)
    "check if node is a class node (now can check both file and headline entries"
    (if (obvz-is-node-headline node)
	    (obvz-is-nodename-cls-node (nth 1 node))
	(obvz-is-nodename-cls-node node)))
    

(defun obvz-is-nodename-cls-node (node)
    """checks if the name of a node is a class (cls_)"""
	    
    (let ((node-state nil))
	(if (> (length node) 4)
		(if (equal (substring node 0 4) "cls_")
			(setq node-state t)))
	node-state))

			
			
	

(defun children-specific-depth-let (node depth)
    "retrieves children and hierarchical links of node $node to level $depth"
    (let (
	  (nodes-upper-level (list node))
	  (all-nodes ())
	  ;; (all-nodes (list node))
	  (all-links ())
	  (temp-res ())
	  (all-res ())
	  )

	;; push node to all nodes when not class node
	(when (not (obvz-is-node-cls-node node))
	    (push node all-nodes))
	
	;; for each depth step
	(dotimes (i depth)

	    ;; temp-res: contains 2 lists (nodes/links) for each upper-level-node
	    ;; idk it's a bit ugly
	    ;; but idk how to work without temp-res
	    (push (mapcar 'loop-over-upper-level-nodes nodes-upper-level) temp-res)

	    ;; push nodes/links to local vars
	    (mapc (lambda (nodex) (push nodex all-nodes)) (obvz-flatten-list (mapcar 'cdr (car temp-res))))
	    (mapc (lambda (linkx) (push linkx all-links)) (obvz-flatten-list (mapcar 'car (car temp-res))))
	    
	    (setq nodes-upper-level (obvz-flatten-list (mapcar 'cdr (car temp-res))))
	    (setq temp-res ())

	    )
	(push all-links all-res)
	(push all-nodes all-res)

	all-res
	 
	)
    )



(defun obvz-children-specific-depth (node depth)
    "retrieves children and hierarchical links of node $node to level $depth"
    (let (
	  (nodes-upper-level (list node))
	  (all-nodes ())
	  ;; (all-nodes (list node))
	  (all-links ())
	  (temp-res ())
	  (all-res ()))


	;; push node to all nodes when not class node
	(when (not (obvz-is-node-cls-node node))
	    (push node all-nodes))
	
	;; for each depth step
	(dotimes (i depth)


	    (setq temp-res (mapcar 'obvz-loop-over-upper-level-nodes nodes-upper-level))
	    (setq temp-nodes (mapcar 'cdr temp-res))
	    (setq temp-nodes2 (mapcar 'car temp-nodes))

	    ;; need to deal with lists of lists, but can't aggressively flatten
	    (mapcar (lambda (i) (mapcar (lambda (k) (push k all-nodes)) i)) temp-nodes2)
	    ;; deal with links
	    (mapc (lambda (linkx) (push linkx all-links)) (obvz-flatten-list (mapcar 'car  temp-res)))
	    
	    ;; (setq nodes-upper-level (obvz-flatten-list (mapcar 'cdr (car temp-res))))
	    (setq nodes-upper-level ())
	    (mapcar (lambda (i) (mapcar (lambda (k) (push k nodes-upper-level)) i)) temp-nodes2)
	    (setq nodes-upper-level2 nodes-upper-level)
	    
	    )
	
	(push all-links all-res)
	(push all-nodes all-res)

	all-res))


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
	    ;; (print node)
	    (setq friends (org-brain-friends node))
	    (while friends
		(setq friend (car friends))
		
		(setq edge-annot (org-brain-get-edge-annotation node friend))
		;; check edge annotation requirement
		(if (equal obvz-only-use-annotated-edges t)
			(progn 
			    (if (not (equal edge-annot nil))
				    (progn
					(push (concat (obvz-get-node-name node) " -- " (obvz-get-node-name friend) " -- " edge-annot) friend-links)
					(push friend friend-nodes))))
		    ;; if requirement nil: use friend regardless of annotation
		    (progn
			(push (concat (obvz-get-node-name node) " -- " (obvz-get-node-name friend) " -- " edge-annot) friend-links)
			(push friend friend-nodes)))

		(setq friends (cdr friends)))

	    (setq nodes (cdr nodes)))
	(push friend-links res)
	(push friend-nodes res)
	;; friend-links
	res))

	

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

	    (setq node-res (obvz-children-specific-depth rel-node 8))
	    
	    ;; (push (car node-res) total-nodes)
	    (mapc (lambda (i) (push i total-nodes)) (car node-res))
	    
	    (push (cdr node-res) total-links)

	    (setq rel-nodes (cdr rel-nodes))
	    
	    )

	;; (print total-nodes)
	;; (setq uniq-nodes (remove-duplicates (flatten-list total-nodes)))
	(setq uniq-nodes (delete-dups total-nodes))

	;; (cl-delete-if (lambda (k) (string-match-p "cls_" k)) uniq-nodes)
	(cl-delete-if (lambda (k) (obvz-is-node-cls-node k)) uniq-nodes)

	;; (print uniq-nodes)
	;; (setq uniq-nodes (cl-delete-if (lambda (k) (string-match-p "cls_" k)) uniq-nodes))
	
	;; handle friend links
	(setq friend-res (obvz-get-friend-links uniq-nodes))
	(setq friend-links (cdr friend-res))
	(push friend-links total-links)
	(setq all-links (obvz-flatten-list total-links))
	(setq link-string (mapconcat 'identity all-links ";"))


	;; handle friend nodes
	(setq friend-nodes (car friend-res))
	(mapc (lambda (i) (push i uniq-nodes)) friend-nodes)
	
	;; (setq uniq-nodes (delete-dups (flatten-list (append uniq-nodes friend-nodes))))
	(delete-dups uniq-nodes)
	
	;; (setq uniq-nodes (remove-duplicates (flatten-list (append uniq-nodes friend-nodes))))
	;; delete cls_nodes from being there if alone
	;; not clear if effective: what if referred to as friend? 
	
	
	;; (message "all nodes there")
	
	;; (setq node-string (mapconcat 'identity uniq-nodes ";"))
	;; include (or not) node texts
	(if (equal obvz-include-node-texts t)
		(progn
		    (setq node-texts (mapcar 'org-brain-text uniq-nodes))
		    (setq node-text-alist (mapcar* #'cons (mapcar 'obvz-get-node-name uniq-nodes) node-texts))
		    )
	    (setq node-text-alist (mapcar* #'cons (mapcar 'obvz-get-node-name uniq-nodes) (make-list (len uniq-nodes) "")))
	    )
	;; (message "node texts there")

	;; get current node
	(if (equal obvz-highlight-current-node t)
		(setq current-node (obvz-get-node-name (org-brain-entry-at-pt)))
	    (setq current-node nil))
	    

	(setq graph-dict
	      `(("links" . ,all-links)
		("nodes" . ,(mapcar 'obvz-get-node-name uniq-nodes))
		("cur_node" . ,current-node)
		("node_texts" . ,node-text-alist)
		("draw_arrow_toggle" . ,obvz-draw-arrow)
		("layout_type" . ,obvz-layout-type)
		)
	      )
	graph-dict
	)
    )

;; data structure: dict with key node, value text
;; is itself dict in graph_dict


(defun obvz-switch-node-text-inclusion()
    "toggle node text inclusion on or off"
    (interactive)
    (if (equal obvz-include-node-texts t)
	    (setq obvz-include-node-texts nil)
	(setq obvz-include-node-texts t)
	)
    (obvz-update-graph)
    )




(defun obvz-reposition-nodes()
    "redraw layout, either soft (apply forces to current layout) or hard (from random starting positions)"
    (interactive)
    (let ((called-prefix current-prefix-arg)
	  (obvz-redraw-alist ())
	  )
	(if (equal called-prefix nil)
		(setq obvz-redraw-alist '(("redraw" . "soft")))
	    (setq obvz-redraw-alist '(("redraw" . "hard")))
	    )
	;; (zmq-send sock (json-encode-alist obvz-redraw-alist))
	(obvz-send-to-python (json-encode-alist obvz-redraw-alist))
	)
    )


(defun obvz-update-graph ()
    "update graph; intended for changed node text and after restarting"
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

(defun obvz-update-graph-hard ()
    "update graph in any case (maybe necessary); intended for changed node text and after restarting"
    (interactive)
    (setq obvz-current-config (obvz-create-graph-dict obvz-include-node-texts))
    (setq obvz-most-recent-config obvz-current-config)
    (obvz-send-to-python (json-encode-alist obvz-current-config))
    )

(defun obvz-start ()
    (interactive)
    (add-hook 'org-brain-after-visualize-hook 'obvz-update-graph)
    (shell-command (mapconcat 'identity `("cd" ,obvz-dir "&& python3.7 obr_viz_server.py" ,obvz-connection-type ,obvz-layout-type "&") " "))
     
    (setq obvz-most-recent-config ()))
    

(defun obvz-stop ()
    (interactive)
    (remove-hook 'org-brain-after-visualize-hook 'obvz-update-graph))
    
    

(defun obvz-export ()
    """exports the currently visualized graph to dot or svg."""
    (interactive)
    (let ((obvz-export-format (completing-read "Set export format: " '("dot" "svg")))
	  (obvz-export-file (expand-file-name (read-file-name "Export file: ")))
	  (export-dict ()))
	  
	(setq export-dict `(("export_type" . ,obvz-export-format)
			    ("export_file" . ,obvz-export-file)))

	(obvz-send-to-python (json-encode `(("export" . ,export-dict))))))
	  
	  
	
	

(defun obvz-set-layout-type ()
    "set the layout algorithm; choices are graphviz dot and a custom force-directed algorithm"
    (interactive)
    (setq obvz-layout-type (completing-read "Set layout type: " '("dot" "force")))
    (obvz-send-to-python (json-encode `(("layout_type" . ,obvz-layout-type))))
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
	
	    

;; (add-hook 'org-brain-after-visualize-hook 'obvz-update-graph) ;; automatic redrawing with org-brain

(setq obvz-highlight-current-node nil)



(setq obvz-connection-type "dbus")

(setq obvz-dir "~/Dropbox/personal_stuff/obr-viz/")
(setq obvz-include-node-texts t)
(setq obvz-only-use-annotated-edges t)
(setq obvz-most-recent-config ())
(setq obvz-draw-arrow t)
(setq obvz-highlight-current-node t)
(setq obvz-layout-type "force")




(define-key org-brain-visualize-mode-map "N" 'obvz-switch-node-text-inclusion)
(define-key org-brain-visualize-mode-map "R" 'obvz-reposition-nodes)
(define-key org-brain-visualize-mode-map "U" 'obvz-update-graph-hard)


;; ============== zmq section ================

;; (require 'zmq)
;; (setq sock (zmq-socket (zmq-context) zmq-PUB))
;; (zmq-bind sock "tcp://127.0.0.1:5556")
;; (setq obvz-connection-type "zmq")
    
