(require 'cl)

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


(defun obvis-create-children-links (parent child)
    """basic hierarchical function"""
    (if (> (length parent) 4)
	    (when (not (equal (substring parent 0 4) "cls_"))
		(concat parent " -- " child " -- isa"))
	(concat parent " -- " child " -- isa"))
    )


(defun loop-over-upper-level-nodes (parent)
    """get nodes and hierarchical links of parent node"""
    ;; needs both nodes and links for some tests later iirc
    (let ((childrenx (org-brain-children parent))
	  (res ()
	       ))
	(push childrenx res)
	
	(push (mapcar (lambda (child)
			  (funcall #'obvis-create-children-links parent child))
		      childrenx) res)
	;; can also push to nodes/links to function-scoped let variables
	res
	))



(defun children-specific-depth-let (node depth)
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

(defun get-friend-links (nodes)
    """get links between nodes"""
    (setq nodes-backup nodes)

    (setq friend-links ())

    (while nodes
	(setq node (car nodes))
	(setq friends (org-brain-friends node))
	(while friends
	    (setq friend (car friends))
	    
	    (setq edge-annot (org-brain-get-edge-annotation node friend))
	    (if (not (equal edge-annot nil))
		    (push (concat node " -- " friend " -- " edge-annot) friend-links)
		)

	    ;; check alphabetic order to avoid duplicates
	    ;; (setq comp (car (cl-sort (list node friend) #'string-lessp)))
	    ;; (if (eq node comp)
	    ;; 	    (progn
	     
	    ;; 		(setq membership-check (member friend nodes-backup))
	    ;; 		(if (> (len membership-check) 0)
	    ;; 			(push (concat node " -- " friend) friend-links)
	    ;; 		    )
	    ;; 		)
	    ;; 	)

	    (setq friends (cdr friends))

	    )
	(setq nodes (cdr nodes))
	)
    friend-links
    )

	

(defun obr-viz ()
    (interactive)
    """main function"""
      ;; (mapcar 'children-specific-depth org-brain-pins 3)
    (setq rel-nodes org-brain-pins)
      
    (setq total-nodes ())
    (setq total-links ())

    
    ;; get hierarchical relations
    ;; just don't push isa relations (cdr node-res) if rel-node == cls to total links? ? 
    (while rel-nodes
	(setq rel-node (car rel-nodes))

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

	(setq node-res (children-specific-depth-let rel-node 8))
	(push (car node-res) total-nodes)
	(push (cdr node-res) total-links)

	(setq rel-nodes (cdr rel-nodes))
	)

    (setq uniq-nodes-prep (counter (flatten-list total-nodes)))
    (setq uniq-nodes (mapcar 'car uniq-nodes-prep))


    (setq friend-links (get-friend-links uniq-nodes))

    (push friend-links total-links)

    (setq all-links (flatten-list total-links))

    (setq node-string (mapconcat 'identity uniq-nodes ";"))
    (setq link-string (mapconcat 'identity all-links ";"))

    ;; (print link-string)

    ;; can just send multiple things to eaf lol
    ;; maybe good for adding groups later or something

    ;; (eaf-setq update_check 0)
    ;; (eaf-setq cur_node (org-brain-entry-at-pt))
    ;; (setq update_check_elisp 0)

    ;; (eaf-setq nodes node-string)
    ;; (eaf-setq links link-string)

    ;; (eaf--send-var-to-python)

    )
    

;; (setq all-string (mapconcat 'identity node-string link-string ""))

;; (mapconcat 'identity '("foo" "bar" "baz") ", ")      


;; watch -n 1 tail file


(defun obr-viz-redraw()
    (interactive)
    """reload current edge list"""
    
    (eaf-setq update_check 1)
    (setq update_check_elisp 1)
    (eaf--send-var-to-python)

    (while (eq update_check_elisp 1)
	(sit-for 0.01)
	)
    (eaf-setq update_check 0)
    (eaf--send-var-to-python)
    (setq update_check_elisp 0)
    )

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




;; send_to_python seems to be called now automatically through eaf-setq
;; 

(define-key org-brain-visualize-mode-map "G" 'obr-viz)
(define-key org-brain-visualize-mode-map "R" 'obr-viz-redraw)







;; (setq sock (zmq-socket (zmq-context) zmq-REQ))
;; (zmq-connect sock "tcp://127.0.0.1:5556")
;; (zmq-send sock "asdf")
;; (zmq-send sock (obr-viz))

;; (setq mes_back (zmq-recv sock))
;; (zmq-connect sock "tcp://127.0.0.1:5556")


(require 'zmq)

(setq sock (zmq-socket (zmq-context) zmq-PUB))
(zmq-bind sock "tcp://127.0.0.1:5556")




(setq time-most-recent-vcall 0)
(defun obr-viz-call ()
    "only call obr-viz if it has not been called in last 1.2 seconds, 
should be changed to not work on time but on changed configuration tho"
    (setq time-now (float-time))
    (setq time-since-last-call (- time-now time-most-recent-vcall))
    (message (number-to-string time-since-last-call))
    (if (> time-since-last-call 1.2)
	    (progn
		(update-obr-viz)
		(setq time-most-recent-vcall time-now)
		)
	)
    )
	  
    
(add-hook 'org-brain-after-visualize-hook 'obr-viz-call)


;; (setq testctr 0)

;; (defun test-funx ()
;;     (setq testctr (1+ testctr))
;;     )


;; (add-hook 'org-brain-visualize-text-hook 'test-funx)
;; (remove-hook 'org-brain-visualize-text-hook 'test-funx)


