; CloGANN: Clojure abstraction for breeding (Recurrent) Neural Networks (NN) with the Genetic Algorithm (GA).
; See the 'population.clj' file for documentation.
; Tomasz Dryjanski ( https://github.com/tomekd789 )

; unsolved:
; - 'polirythmic' breeding

(ns clogann.core
  (:gen-class))

; These will be taken from the 'population.clj' file...
(declare initial-vector take-next-sample calculate-final-evaluation params population)

; ... here:
(defn load-population
"Reads the population from file, and defines parameters"
[]
  (load-file
    "population.clj")
  (def population-save-interval
    (first (:population-save-interval params)))
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
  (def popul
    (if initialize-population
      ; Population filled with 0.0s, evaluations set to default-null-eval
      (into [] (take population-size (repeat [(into [] (take network-size
         (repeat (into [] (take network-size (repeat 0.0)))))) default-null-eval])))
      population))
  (def immutable-file-part
    (first (clojure.string/split (slurp "population.clj") #"\n\n;;; The mutable part\n"))))

(defn save-new-population-file
"Saves updated population file after some breeding, with altered params"
  [mutation-probability-inverse, generation, parallelism, popul]
  (let [fname (str "population_" generation ".clj")
        write #(spit fname % :append true)]
[(spit fname "") ; Touch file
 (write immutable-file-part)
 (write "\n\n;;; The mutable part\n")
; The subsequent body can be expressed more concisely using e.g. Pretty Print.
; It was consciously avoided to reduce dependencies, it's up to the user to do any refactoring.
 (write "(def params\n{\n")
 ; There is a low-priority plan to replace the following by a Pretty Print invocation
 (write ":population-save-interval '(10 \"New file is saved and evals displayed every % generation; integer\")\n")
 (write (str ":mutation-probability-inverse '(" mutation-probability-inverse " \"Self-adjustable. P. of weight modification when a new org is created; integer\")\n"))
 (write ":crossover-probability '(0.4 \"...when a new org is created; double\")\n")
 (write ":initialize-population '(false \"If true, a new, zeroed population will be created; boolean\")\n")
 (write ":default-null-eval '(0.0 \"Initial evaluation taken if initialize-population; double\")\n")
 (write ":iterations-per-input '(1 \"Network cycles per each value provided; integer\")\n")
 (write ":population-size '(50 \"The population size; integer\")\n")
 (write ":network-size '(12 \"The network size; integer\")\n")
 (write (str ":generation '(" generation " \"Current generation; integer\")\n"))
 (write (str ":parallelism '(" parallelism " \"# of organisms evaluated in parallel - if 0 then # of the JVM cores; integer\")\n"))
 (write "})\n\n")
 (write "; The population itself; made of pairs: evaluation, and an organism\n")
 (write (str "(def population " popul ")"))]))

(defn print-top-5-evaluates []
  (clojure.string/join ";  " (take 5 (map last popul)))
)

(defn mul-vector-array"Multiply a transposed vector by a row-arranged array"
[vector-transposed array-by-rows]
(into [] (map
          #(reduce + (map * vector-transposed %)) 
          array-by-rows)))

(defn -main
"The main function"
[& args]
(load-population))
