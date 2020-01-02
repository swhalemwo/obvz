(require 'cl)

(defun children-specific-depth (node depth)
    """get nodes that are there"""
    ;; would be nice to also already get links: hierarchical links are here
    (setq nodes-upper-level ())
    (push node nodes-upper-level)

    (setq all-nodes ())
    (push node all-nodes)
    
    (setq all-links ())
    
    ;; loop over levels
    (while (> depth 0)
	;; (print depth)

	(setq children-nodes ())
	;; loop over nodes at level
	(while nodes-upper-level
	    (setq node-upper-level (car nodes-upper-level))
	    (setq children (org-brain-children node-upper-level))

	    ;; (print node-upper-level)
	    ;; (print children)
	    (push children all-nodes)
	    (push children children-nodes)
	    
	    ;; loop over children to set specific links
	    (while children
		(setq child (car children))
		(setq link (concat node-upper-level " -- " child " -- isa"))
		(push link all-links)

		(setq children (cdr children))
		)
	    ;; (setq children-nodes (flatten-list children))
	    (setq nodes-upper-level (cdr nodes-upper-level))
	    )
	
	;; (print "setting next level")
	;; (print children-nodes)
	(setq nodes-upper-level (flatten-list children-nodes))
    
	(setq depth (- depth 1))
    
	)
    (setq all-nodes (flatten-list all-nodes))

    (setq returns ())
    (push all-links returns)
    (push all-nodes returns)
    
    returns
    )

(defun obvis-create-children-links (parent child)
    (concat parent " -- " child " -- isa"))


;; (mapcar 'obvis-create-children-links children)


;; (defvar packages '(("auto-complete" . "auto-complete")
;;                    ("defunkt" . "markdown-mode")))

;; (defvar packages2 '("auto-complete" "markdown-mode"))


(defun toy-fnx (author name)
    "Just testing."
    (message "Package %s by author %s" name author)
    ;; (print
    (concat "Package " name " by author " author)
    ;; )
    ;; (sit-for 1)
    )

;; (mapcar (lambda (package)
;;           (funcall #'toy-fnx (car package) (cdr package)))
;;         packages)


;; (mapcar (lambda (package)
;;           (funcall #'toy-fnx "some_guy" (car package)))
;;         packages2)


;; (mapcar (lambda (package)
;;           (funcall #'toy-fnx "some_guy" package))
;;         packages2)






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
	(if (equal (substring rel-node 0 4) "cls_")
		(progn
		    (setq node-res (children-specific-depth rel-node 8))
		    (push (car node-res) total-nodes)
		    )
	    (progn
		(setq node-res (children-specific-depth rel-node 8))
		(push (car node-res) total-nodes)
		(push (cdr node-res) total-links)
		)
	    )
	;; (print "total nodes")
	;; (print total-nodes)
	(setq rel-nodes (cdr rel-nodes))
	)

    (setq uniq-nodes-prep (counter (flatten-list total-nodes)))
    (setq uniq-nodes (mapcar 'car uniq-nodes-prep))


    (setq friend-links (get-friend-links uniq-nodes))

    (push friend-links total-links)

    (setq all-links (flatten-list total-links))

    (setq node-string (mapconcat 'identity uniq-nodes ";"))
    (setq link-string (mapconcat 'identity all-links ";"))

    ;; can just send multiple things to eaf lol
    ;; maybe good for adding groups later or something

    (eaf-setq update_check 0)
    (eaf-setq cur_node (org-brain-entry-at-pt))
    (setq update_check_elisp 0)

    (eaf-setq nodes node-string)
    (eaf-setq links link-string)

    (eaf--send-var-to-python)

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

  (eaf-open "eaf-obr-test" "obr-test")
  (eaf--send-var-to-python)
  )

(define-key org-brain-visualize-mode-map "G" 'obr-viz)
(define-key org-brain-visualize-mode-map "R" 'obr-viz-redraw)





    
