; See the 'population.clj' file for documentation.
; Tomasz Dryjanski ( https://github.com/tomekd789 )

(ns clogann.core
  (:gen-class))

; These will be taken from the 'population.clj' file...
(declare initial-vector take-next-sample calculate-final-evaluation params population)

; ... here:
(defn load-population
"Reads the population from file, and defines parameters"
[]
  (load-file "population.clj")
  (def population-save-interval
    (first (:population-save-interval params)))
  (def population-save-folder
    (first (:population-save-folder params)))
  (def mutation-probability-inverse
    (first (:mutation-probability-inverse params)))
  (def crossover-probability
    (first (:crossover-probability params)))
  (def initialize-population
    (first (:initialize-population params)))
  (def default-null-eval
    (first (:default-null-eval params)))
  (def iterations-per-input
    (first (:iterations-per-input params)))
  (def population-size
    (first (:population-size params)))
  (def network-size
    (first (:network-size params)))
  (def generation
    (first (:generation params)))
  (def parallelism
    (let [p (first (:parallelism params))]
      (if (= p 0)
        (.availableProcessors (Runtime/getRuntime))
        p)))
  (def initial-population
    (if initialize-population
      ; Population filled with 0.0s, evaluations set to default-null-eval
      (into [] (take population-size (repeat [(into [] (take network-size
         (repeat (into [] (take network-size (repeat 0.0)))))) default-null-eval])))
      population))
  (def immutable-file-part
    (first (clojure.string/split (slurp "population.clj") #"\n\n;;; The mutable part\n"))))

(defn save-new-population-file
"Saves updated population file after some breeding, with altered params"
[mutation-probability-inverse generation population]
(let [fname (str population-save-folder "population_" generation ".clj")
        write #(spit fname % :append true)]
[(spit fname "") ; Touch file
 (write immutable-file-part)
 (write "\n\n;;; The mutable part\n")
 ; The subsequent body can be expressed more concisely using e.g. Pretty Print.
 ; It was consciously avoided to reduce dependencies, it's up to the user to do any refactoring.
 (write "(def params\n{\n")
 (write (str ":population-save-interval '(" population-save-interval
             " \"New file is saved and evals displayed every % generation; integer\")\n"))
 (write (str ":population-save-folder '(" population-save-folder
             " \"New files are saved to this folder\")\n"))
 (write (str ":mutation-probability-inverse '(" mutation-probability-inverse
    " \"Self-adjustable. P. of a weight modification when a new org is created; integer\")\n"))
 (write (str ":crossover-probability '(" crossover-probability
             " \"...when a new org is created; double\")\n"))
 (write ":initialize-population '(false \"If true, a new, zeroed population will be created; boolean\")\n")
 (write (str ":default-null-eval '(" default-null-eval
             " \"Initial evaluation taken if initialize-population; double\")\n"))
 (write (str ":iterations-per-input '(" iterations-per-input
             " \"Network cycles per each value provided; integer\")\n"))
 (write (str ":population-size '(" population-size
             " \"The population size; integer\")\n"))
 (write (str ":network-size '(" network-size " \"The network size; integer\")\n"))
 (write (str ":generation '(" generation " \"Current generation; integer\")\n"))
 (write (str ":parallelism '(" parallelism
    " \"# of organisms evaluated in parallel - if 0 then # of the JVM cores; integer\")\n"))
 (write "})\n\n")
 (write "; The population itself; made of vectors [organism evaluation]\n")
 (write (str "(def population " population ")"))]))

(defn mul-vector-array
"Multiply a transposed vector by a row-arranged array"
[vector-transposed array-by-rows]
(into [] (map
          #(reduce + (map * vector-transposed %)) 
          array-by-rows)))

(defn network-iteration
"Change the state vector by a single network iteration"
[state-vector network]
(into [] (map #(max % 0.0) (mul-vector-array state-vector network))))

;TBC: sample evaluation here

(defn evaluate-network
"Evaluates a network and returns it with the updated evaluation"
[organism] (assoc organism (dec (count organism)) default-null-eval))
;TBC

(defn calculate-evaluations
"Calculate evaluations for a given population"
([population] (calculate-evaluations [] population))
([processed-vector remaining-vector]
 (if (empty? remaining-vector)
   processed-vector
   (recur (concat processed-vector
                  (pmap evaluate-network (take parallelism remaining-vector)))
          (drop parallelism remaining-vector)))))

(defn crossover
"Applies crossing-over to a population"
[population]
population)
;TBC

(defn mutate
"Enters random mutations to a population"
[population pmi]
population
)
;TBC

(defn breed-new-population
  "Takes a population, and breeds a new one; also informs about new organisms count"
  [population pmi]
  (let [unsorted-children (calculate-evaluations (mutate (crossover population) pmi))
        new-population (take population-size (sort-by last > (concat unsorted-children population)))
        ; as per the documentation, clojure.core/sort is 'conservative', or 'stable'
        ; i.e. children go first if evaluations are equal
        least-new-evaluation (last (last new-population))
        count-those-with-least-eval
          (fn [population] (count (filter #(= (last %) least-new-evaluation) population)))]
    [new-population
     (+ (count (filter #(> (last %) least-new-evaluation) unsorted-children))
        (min (count-those-with-least-eval unsorted-children)
             (count-those-with-least-eval new-population)))]))

(defn print-top-5-evaluations [generation population]
  "Print the generation number and  top 5 evaluations in the population"
  ; The only function with side effects
                                        ; Leaving up to the user to add e.g. (format "%.3f" %) in between
  (println (str "Generation: " generation "; Top 5 evaluations: "
                (clojure.string/join ", " (take 5 (map last population))))))

(def pmi-update-frequency 10)
; Mutation probability will be updated every % generations
; It's a result of the author's experience, but the user may decide alter it

(defn the-main-loop
"The main clogann loop, called once from main, infinite"
; It breeds next populations iteratively.
; Every population-save-interval the population is saved to a new file.
; Every pmi-upgrade-frequency the mutation probability is upgraded,
; based on the number of new organisms entering the population in the meantime.
[]
(loop [current-generation generation
       current-pmi mutation-probability-inverse
       current-population initial-population
       accu-new-org-count 0]
  (if (= 0 (mod current-generation population-save-interval)) ; i.e. every save-interval
    (do
      (save-new-population-file current-pmi current-generation current-population)
      (print-top-5-evaluations current-generation current-population)))
  (let [[new-population new-org-count] (breed-new-population current-population current-pmi)
        update-pmi? (= 0 (mod current-generation pmi-update-frequency))]
    (recur
     (inc current-generation)
     (if update-pmi? ; with pmi-upgrade-frequency,
       (if (> accu-new-org-count population-size) ; if too many new organisms created in the meantime,
         (max (dec current-pmi) 5) ; increase mutation probability (i.e. dec inverse), but not above 1/5
         (inc current-pmi)) ; otherwise decrease the mutation probability (i.e. inc inverse)
       current-pmi)
     new-population ; take the new population for the next main loop iteration
     (if update-pmi? ; if the accu-new-org-count has been "consumed" above, start building a new one
       new-org-count
       (+ new-org-count accu-new-org-count))))))

(defn -main
"The main function"
[& args]
(load-population)
(the-main-loop))
