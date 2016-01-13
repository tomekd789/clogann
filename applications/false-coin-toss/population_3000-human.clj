; CloGANN: Clojure abstraction for breeding (Recurrent) Neural Networks (NN) with a Genetic Algorithm (GA).

; Here it is applied to check "human pseudorandom generator" predictability.
; It takes the coin-tosses.dat file and breeds a network capable of predicting next values in the sequence.
; The fit function is a percentage of tosses successfully guessed.
; (pseudo-random: (spit coin-tosses.dat (repeatedly 

; The file coin-tosses.dat is expected to be a stream of #{\0 \1} characters,
; terminated by any non-alphanumeric character. It's now converted to a vector of int 0s and 1s:
(def input-seq (into [] (map #(- (int %) 48) (slurp "coin-tosses.dat"))))
(def input-seq-count (count input-seq))

(def initial-vector [false]) ; Flag to stop - will change to true on next evaluation

(defn take-next-sample
"Generate the single sample"
  [user-vector]
  [(first user-vector) ; Only one sample is going to be taken,
   [true] ; ...next round the flag will be 'true'
   [0] ; Position pointer for the input-seq.
   (fn [state-vector input-vector] ; The provide-input function
       [(into [] (concat
         [(get input-seq (first input-vector)) ; The next input-seq item.
            1.0 ; A network bias.
            (rand)] ; Some noise to make it a Boltzmann network.
         (subvec state-vector 3))) ; The rest of the state-vector.
       [(inc (first input-vector))]])  ; Move the pointer forward.
    [1 0]; [Position pointer for the input-seq shifted one ahead, Count of tosses correctly guessed]
    (fn [state-vector user-vector] ; The interpret-output function
        (let [input-pointer (first user-vector)
              actual-next-item (get input-seq input-pointer)
              terminate? (< actual-next-item 0) ; The terminating value is any below 0
              network-guess (last state-vector) ; Take the last neuron's output as the NN output
              guessed? (or (= (double actual-next-item) network-guess) ; If both are equal, it's a guess
                           (and (= actual-next-item 1) (> network-guess 0)))
                           ; network-guess > 0 is treated as network-guess = 1
              prev-guess-count (second user-vector)
              new-guess-count (+ prev-guess-count (if guessed? 1 0))]
           [terminate?
            [(inc input-pointer) new-guess-count]
            (int (/ (* 100.0 new-guess-count) input-seq-count))]))])

(defn calculate-final-evaluation
"Calculate the final evaluation from the single partial"
  [partial-evaluations]
  (first partial-evaluations)) ; It's a single number vector; converted to a number.

;;; The mutable part
(def params
{
:population-save-interval '(10 "New file is saved and evals displayed every % generation; integer")
:population-save-folder '("human-random/" "New files are saved to this folder")
:mutation-probability-inverse '(201 "Self-adjustable. 1/P of a weight modification when a new org is created; integer")
:crossover-probability '(0.4 "...when a new org is created; double")
:initialize-population '(false "If true, a new, zeroed population will be created; boolean")
:default-null-eval '(0.0 "Initial evaluation taken if initialize-population; double")
:iterations-per-input '(1 "Network cycles per each value provided; integer")
:population-size '(50 "The population size; integer")
:network-size '(6 "The network size; integer")
:generation '(3000 "Current generation; integer")
:parallelism '(true "True - with pmap; False - with map")
})

; The population itself; made of vectors [network its-evaluation]
(def population [[[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 89.0 -1.0] [1.125 7.5 3.5 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 3.0 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[4.5 -26.0 86.0 -0.5 5.0 -3.0] [2.125 9.5 0.125 -16.5 -0.25 -2.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-233.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [2.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 89.0 -1.0] [1.125 7.5 3.5 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-116.0 3.75 -0.8125 -0.25 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 89.0 -1.0] [1.125 7.5 3.5 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-116.0 3.75 -0.8125 -0.25 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 3.0 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -1.0] [9.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 7.0 -36.0 -0.5 -2.0] [4.25 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[0.75 -12.0 41.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 21.0 -1.5 44.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -2.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 6.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 7.0 -36.0 -1.0 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [2.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 89.0 -1.0] [1.125 7.5 3.5 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.25 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[4.5 -26.0 86.0 -0.5 5.0 -3.0] [2.125 9.5 0.125 -16.5 -0.25 -2.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-233.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 44.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -2.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 6.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[1.75 -12.0 43.0 -1.5 4.0 -2.5] [2.125 3.75 1.625 -16.5 -0.25 -1.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-115.5 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 0.5625 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[0.75 -12.0 41.0 -1.5 5.0 -3.0] [2.125 9.5 1.25 -17.5 -0.25 -2.0] [7.5 4.125 0.54296875 9.0 6.0 2.0] [-224.0 4.0 1.0625 0.125 0.0 0.0] [33.0 4.75 -0.0546875 2.5 0.0 0.0] [-22.0 3.0 -1.5 -1.0 0.5 0.0]] 70] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 70] [[[4.5 -26.0 86.0 -0.5 5.0 -3.0] [2.125 9.5 0.125 -16.5 -0.25 -2.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-233.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[1.75 -12.0 22.0 -2.5 89.0 -1.0] [0.125 7.5 3.5 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.25 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 4.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 6.5 7.0 -36.0 -1.0 -1.0] [8.5 4.25 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-22.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 42.0 -1.5 20.0 -2.0] [1.125 7.5 3.5 -35.0 -0.5 -2.0] [8.5 4.25 5.34375 9.0 1.5 3.0] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[3.5 -12.0 22.0 -1.5 22.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [8.5 8.5 5.34375 9.0 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [2.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 -0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69] [[[1.75 -12.0 22.0 -1.5 88.0 -1.0] [1.125 7.5 7.0 -36.0 -0.5 -1.0] [9.5 4.25 5.34375 4.5 2.5 1.5] [-232.0 3.75 -0.8125 0.5 0.0 0.0] [33.0 2.75 -0.4375 3.0 0.0 0.0] [-21.0 3.0 -3.0 -1.0 0.5 0.0]] 69]])