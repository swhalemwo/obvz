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
    "retrieves children and hierarchical links of $node to level $depth"
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
		(if (not (equal edge-annot nil))
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
	  (uniq-nodes-hierarchy ())
	  (friend-res ())
	  (friend-links ())
	  (friend-nodes ())
	  (all-links ())
	  (link-string ())
	  (node-texts ())
	  (node-text-alist ())
	  (graph-dict ())
	  )
	;; get hierarchical relations

	(while rel-nodes
	    (setq rel-node (car rel-nodes))
	    
	    (setq node-res (children-specific-depth-let rel-node 8))
	    (push (car node-res) total-nodes)
	    (push (cdr node-res) total-links)

	    (setq rel-nodes (cdr rel-nodes))
	    )
	;; (
	(message "children there")

	;; (setq uniq-nodes-prep (counter (flatten-list total-nodes)))
	;; (setq uniq-nodes (mapcar 'car uniq-nodes-prep))

	(setq uniq-nodes (remove-duplicates (flatten-list total-nodes)))

	;; handle links
	(setq friend-res (obvz-get-friend-links uniq-nodes))
	(setq friend-links (cdr friend-res))
	(push friend-links total-links)
	(setq all-links (flatten-list total-links))
	(setq link-string (mapconcat 'identity all-links ";"))
	(message "all links there")

	;; handle nodes
	(setq friend-nodes (car friend-res))
	(setq uniq-nodes (remove-duplicates (flatten-list (list uniq-nodes friend-nodes))))
	(message "all nodes there")
	
	;; (setq node-string (mapconcat 'identity uniq-nodes ";"))
	;; include (or not) node texts
	(if (equal obvz-include-node-texts t)
		(progn
		    (setq node-texts (mapcar 'org-brain-text uniq-nodes))
		    (setq node-text-alist (mapcar* #'cons uniq-nodes node-texts))
		    )
	    (setq node-text-alist (mapcar* #'cons uniq-nodes (make-list (len uniq-nodes) "")))
	    )
	(message "node texts there")

	(setq graph-dict
	      `(("links" . ,all-links)
		("cur_node" . ,(org-brain-entry-at-pt))
		("node_texts" . ,node-text-alist)
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
	(zmq-send sock (json-encode-alist obvz-redraw-alist))
	)
    )




(defun obvz-update-graph ()
    (interactive)

    (setq obvz-current-config (obvz-create-graph-dict obvz-include-node-texts))
    (if (not (equal obvz-current-config obvz-most-recent-config))
	    (progn
		(setq obvz-most-recent-config obvz-current-config)
		(zmq-send sock (json-encode-alist obvz-current-config))
		)
	)
    )



(add-hook 'org-brain-after-visualize-hook 'obvz-update-graph) ;; automatic redrawing with org-brain change

(setq obvz-include-node-texts t)
(setq obvz-most-recent-config ())

(define-key org-brain-visualize-mode-map "N" 'obvz-switch-node-text-inclusion)
(define-key org-brain-visualize-mode-map "R" 'obvz-reposition-nodes)
(define-key org-brain-visualize-mode-map "U" 'obvz-update-graph)


(setq sock (zmq-socket (zmq-context) zmq-PUB))
(zmq-bind sock "tcp://127.0.0.1:5556")

    

;; * eaf sections
;; for now use zmq for easier debugging
;; send_to_python seems to be called now automatically through eaf-setq


(defun eaf-obr-test ()
  "Open EAF demo screen to verify that EAF is working properly."
  (interactive)
  (eaf-setq update_check 0)
  (setq update_check_elisp 0)

  
  (eaf-setq links (obr-viz))
  (eaf-setq update_check 0)
  ;; (eaf-setq cur_node (org-brain-entry-at-pt))
  (setq update_check_elisp 0)

  (eaf-open "eaf-obr-test" "obr-test")

  ;; (eaf--send-var-to-python)
  )


(defun eaf-obr-test2 ()
  "Open EAF demo screen to verify that EAF is working properly."
  (interactive)
  (eaf-setq update_check 0)
  (setq update_check_elisp 0)

    (eaf-setq links (obr-viz))
  (eaf-setq update_check 0)
  ;; (eaf-setq cur_node (org-brain-entry-at-pt))
  (setq update_check_elisp 0)

  ;; (eaf--send-var-to-python)
  )


;; old redraw function
;; (defun obr-viz-redraw()
;;     (interactive)
;;     """reload current edge list"""
    
;;     (eaf-setq update_check 1)
;;     (setq update_check_elisp 1)
;;     (eaf--send-var-to-python)

;;     (while (eq update_check_elisp 1)
;; 	(sit-for 0.01)
;; 	)
;;     (eaf-setq update_check 0)
;;     (eaf--send-var-to-python)
;;     (setq update_check_elisp 0)
;;     )


;; * scrap


;; (remove-hook 'org-brain-after-visualize-hook 'update-obr-viz)
;; (remove-hook 'org-brain-after-visualize-hook 'test-f)


;; (add-hook 'org-brain-after-visualize-hook 'obr-viz-call)
;; (remove-hook 'org-brain-after-visualize-hook 'obr-viz-call)


;; (setq testctr 0)

;; (defun test-funx ()
;;     (setq testctr (1+ testctr))
;;     )


;; (add-hook 'org-brain-visualize-text-hook 'test-funx)
;; (remove-hook 'org-brain-visualize-text-hook 'obr-viz-call)
;; (remove-hook 'org-brain-visualize-text-hook 'test-f2)



;; (defun obr-viz-call ()
;;     "only call obr-viz if it has not been called in last 1.2 seconds, 
;; should be changed to not work on time but on changed configuration tho"

;;     (setq current-pins (mapconcat 'identity org-brain-pins ";"))
;;     (setq current-config (concat current-pins ";" (org-brain-entry-at-pt)))

;;     (if (not (equal current-config most-recent-config))
;; 	    (progn
;; 		(setq current-pins (mapconcat 'identity org-brain-pins ";"))
;; 		(setq most-recent-config (concat current-pins ";" (org-brain-entry-at-pt)))
		
;; 		(message (concat (number-to-string (float-time)) current-config))
;; 		(update-obr-viz)
;; 		)
;; 	)
	
;;     )

;; (print link-string)

;; can just send multiple things to eaf lol
;; maybe good for adding groups later or something

;; (eaf-setq update_check 0)
;; (eaf-setq cur_node (org-brain-entry-at-pt))
;; (setq update_check_elisp 0)

;; (eaf-setq nodes node-string)
;; (eaf-setq links link-string)

;; (eaf--send-var-to-python)

;; old getting children function
;; (defun children-specific-depth (node depth)
;;     """get nodes that are there"""
;;     ;; would be nice to also already get links: hierarchical links are here
;;     (setq nodes-upper-level ())
;;     (push node nodes-upper-level)

;;     (setq all-nodes ())
;;     (push node all-nodes)
    
;;     (setq all-links ())
    
;;     ;; loop over levels
;;     (while (> depth 0)
;; 	;; (print depth)

;; 	(setq children-nodes ())
;; 	;; loop over nodes at level
;; 	(while nodes-upper-level
;; 	    (setq node-upper-level (car nodes-upper-level))
;; 	    (setq children (org-brain-children node-upper-level))

;; 	    ;; (print node-upper-level)
;; 	    ;; (print children)
;; 	    (push children all-nodes)
;; 	    (push children children-nodes)
	    
;; 	    (setq linkss (mapcar (lambda (child)
;; 			(funcall #'obvis-create-children-links node-upper-level child))
;; 				 children))
;; 	    (push linkss all-links)
	    
;; 	    ;; (setq children-nodes (flatten-list children))
;; 	    (setq nodes-upper-level (cdr nodes-upper-level))
;; 	    )
	
;; 	;; (print "setting next level")
;; 	;; (print children-nodes)
;; 	(setq nodes-upper-level (flatten-list children-nodes))
    
;; 	(setq depth (- depth 1))
    
;; 	)
;;     (setq all-links (flatten-list all-links))
;;     (setq all-nodes (flatten-list all-nodes))

;;     (setq returns ())
;;     (push all-links returns)
;;     (push all-nodes returns)
    
;;     returns
;;     )

;; ** some old code which i don't get anymore, probably testting or stuff

;; (setq res-list ())
;; (mapc (lambda (child) (push child res-list)) '(a b c d))


;; return object  of let block has to be last argument in it (not outside of it)

	

;; have to get childrenx as input to  mapcar func
    
;; (mapcar loop-over-upper-level-nodes nodes-upper-level)

;; block to put into function and then mapcar over
;; (while nodes-upper-level
;;     (setq node-upper-level (car nodes-upper-level))
;;     (setq children (org-brain-children node-upper-level))

;;     ;; (print node-upper-level)
;;     ;; (print children)
;;     (push children all-nodes)
;;     (push children children-nodes)
    
;;     (setq linkss (mapcar (lambda (child)
;; 			     (funcall #'obvis-create-children-links node-upper-level child))
;; 			 children))
;;     (push linkss all-links)
    
;;     ;; (setq children-nodes (flatten-list children))
;;     (setq nodes-upper-level (cdr nodes-upper-level))
;;     )


;; (mapcar 'obvis-create-children-links children)


;; (defvar packages '(("auto-complete" . "auto-complete")
;;                    ("defunkt" . "markdown-mode")))

;; (setq node "cls-ppl")

;; (defvar packages2 '("auto-complete" "markdown-mode"))


;; (defun toy-fnx (author name)
;;     "Just testing."
;;     (message "Package %s by author %s" name author)
;;     ;; (print
;;     (concat "Package " name " by author " author)
;;     ;; )
;;     ;; (sit-for 1)
;;     )


;; (mapcar (lambda (package)
;;           (funcall #'toy-fnx (car package) (cdr package)))
;;         packages)


;; (mapcar (lambda (package)
;;           (funcall #'toy-fnx "some_guy" package))
;;         packages2)

;; (setq children (org-brain-children "cls_ppl"))







;; (defun get-friend-links (nodes)
;;     """get links between nodes"""
;;     (setq nodes-backup nodes)

;;     (setq friend-links ())

;;     (while nodes
;; 	(setq node (car nodes))
;; 	(setq friends (org-brain-friends node))
;; 	(while friends
;; 	    (setq friend (car friends))
	    
;; 	    ;; check alphabetic order to avoid duplicates
;; 	    (setq comp (car (cl-sort (list node friend) #'string-lessp)))
;; 	    (if (eq node comp)
;; 		    (progn
	     
;; 			(setq membership-check (member friend nodes-backup))
;; 			(if (> (len membership-check) 0)
;; 				(push (concat node " -- " friend) friend-links)
;; 			    )
;; 			)
;; 		)

;; 	    (setq friends (cdr friends))

;; 	    )
;; 	(setq nodes (cdr nodes))
;; 	)
;;     friend-links
;;     )

;; ** old way of filtering out class membership relations, now works in low-level ((concat parent child)) function
;; of classes, don' get children relationships
;; (if (equal (substring rel-node 0 4) "cls_")
;; 	(progn
;; 	    (setq node-res (children-specific-depth-let rel-node 8))
;; 	    (push (car node-res) total-nodes)

;; 	    )
;;     (progn
;; 	(setq node-res (children-specific-depth-let rel-node 8))
;; 	(push (car node-res) total-nodes)
;; 	(push (cdr node-res) total-links)
;; 	)
;;     )
;; (print "total nodes")
;; (print total-nodes)


;; ** check alphabetic order to avoid duplicates, now replaced by directed edges
;; (setq comp (car (cl-sort (list node friend) #'string-lessp)))
;; (if (eq node comp)
;; 	    (progn

;; 		(setq membership-check (member friend nodes-backup))
;; 		(if (> (len membership-check) 0)
;; 			(push (concat node " -- " friend) friend-links)
;; 		    )
;; 		)
;; 	)

(defun scope-test ()
    (let ((x 1))
	(setq some-var x)
	))

;; ** zmq communication tests

;; (setq sock (zmq-socket (zmq-context) zmq-REQ))
;; (zmq-connect sock "tcp://127.0.0.1:5556")
;; (zmq-send sock "asdf")
;; (zmq-send sock (obr-viz))

;; (setq mes_back (zmq-recv sock))
;; (zmq-connect sock "tcp://127.0.0.1:5556")
