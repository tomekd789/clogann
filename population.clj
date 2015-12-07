; CloGANN: Clojure abstraction for breeding (Recurrent) Neural Networks (NN) with the Genetic Algorithm (GA).
;
;ABOUT THIS PARAMETER FILE
; The purpose of this file is fourfold: It contains a concise description of the project; Together with
; the core.clj file this is all what is needed for the program to run; It contains all necessary user-defined
; paramaters; It creates own copies iteractively with updated values, hence becoming the program output.
;
;GENERAL DESCRIPTION
; This is a multi-layer abstraction for breeding recurrent neural networks capable of solving user-defined
; problems with a genetic algorithm. It can be treated as some kind of indirect programming, i.e. breeding
; networks as programs with heuristic methods.
; Here go the layers, bottom up:
;A NEURON
; A logical entity containing a vector of constant weights. It works by taking inputs from other neurons,
; multiplying them by the weights respectively, summing the multiplies, and returning the result as its output.
; If the number is negative, the returned output is set to zero.
;A NEURAL NETWORK
; Also named an /organism/, or a just /network/, this is a set of 'network-size' neurons (the 'network-size'
; parameter, as well as others mentioned here, are defined in this file in the bottom section). They are all
; connected to one another, output to input, hence the network is Recurrent, or RNN
; ( https://en.wikipedia.org/wiki/Recurrent_neural_network ). All neurons trigger together, so we can talk
; about a network state at a given time. Algebraically, and by implementation, this is an iterative multiplication
; of a network state vector (transposed) by a square array of weights (row-oriented), and substituting
; all negative values by zeroes subsequently.
;A SAMPLE
; A network can be also seen as a function processing its state vector iteractively, in the way described above.
; At the beginning, and then every 'iterations-per-input', the state vector is additionaly processed by
; a user-defined function for providing input to the network. It just takes and returns the state vector, altered.
; It is up to the user to decide how many neurons will be impacted, and how many times. Conceputally, this is
; treating some neurons as input, i.e. disregarding network outputs for them and providing the user defined
; value instead. Similarly, the state vector is then processed by another user-defined function to interpret
; the network output. It is expected to return a boolean flag interpreted as a stop signal, and an evaluation
; value. The described network run until stop is called a /sample/, and the return value is a /partial evaluation/.
;NETWORK EVALUATION
; A network can be sampled many times, returning many partial evaluations saved as a list. This is also controlled
; by a user-defined 'take-next-sample function capable of providing a higher level stop signal (this is not to be
; confused with the stop signal for a sample - this time its for a full, multiple-samples network evaluation).
; Then the partial evaluations vector is taken as the argument for the user-defined 'calculate-final-evaluation'
; function, returning a single value taken as the /network evaluation/.
;A POPULATION
; This is a population-size vector of organisms paired with their evaluations.
; If 'initialize-population' is 'true', the 'population' definition is disregarded, and a new 0.0-filled
; population is created, with evaluations initially set to 'default-null-eval'.
;THE GENETIC ALGORITHM
; Conceptually, core.clj works as follows:
; - A new pool of organisms is created from the existing one,
; - The pool is processed by a cross-over, with 'crossover-probability',
; - All its weights are then subject to a mutation, with (/ 1 mutation-probability-inverse) probability;
;   a mutation can be #(inc %), #(dec %), #(/ % 2), or #(* % 2) - all equally probable,
; - All organisms in the new pool are then evaluated, with use of 'pmap' parallelism. The degree
;   of parallelism is defined as 'parallelism'. If 0 is provided then the JVM CPU count is taken,
;   and then written explicitly to subsequent output files,
; - The extended population is then sorted by its evaluations, descending (i.e. to maximize the
;   evaluation result), counter-conservative (i.e. new organisms are preferred), and 'population-size'
;   organisms are taken as the new population.
;   The Genetic Algorithm (GA) described here tends to _maximize_ the evaluation function value.
;THE PROGRAM
; Every 'population-save-interval'-th generation, the population is saved to a new file,
; with the name indexed by the generation number. User can assume that this is the only side effect
; introduced by core.clj. User needs to be aware of usage of 'pmap', e.g. no interaction
; is expected to happen between organisms when they are evaluated, and one needs to be careful
; with any side effects, resulting e.g. in locks.
; The 'mutation-probability-inverse' is updated every 10-th generation so that the population
; is more or less entirely replaced over such a life span.
;
;YET ABOUT THIS FILE
; core.clj looks for the file, named 'population.clj', and starts by executing it using 'load-file'. The original
; population file is not referenced further, and it can be removed or altered e.g. for another processing.
; core.clj then works iteratively by saving subsequent copies of the population file, with updated content,
; in the folder defined by the user.
; The 'population.clj' file consists of two sections:
; - The first one goes from the top since the '';;; The mutable part'' line, and is copied without any changes.
;   It must contain necessary definitions for core.clj.
; - The second part is a snapshot dump of all parameters (some of them alter), and the population itself



;DEFINITIONS
; For educational purposes this file defines a working example, breeding a network capable of factorizing
; natural numbers, i.e. for a given k it returns m, n such that k = m*n. m and n do not have to be prime numbers.

; 'initial-vector': This is the initial value taken as an argument for 'take-next-sample' on the first run
(def initial-vector [100]) ; In this particular example the vector contains just a decrementing counter

; 'take-next-sample': essentially it transforms the initial vector subsequently, also
; returing values necessary for a sample evaluation to work.
; Its parameter is a user-defined vector; initially the 'initial-vector' is taken.
; The function has to return a vector consisted of the following:
; - A boolean flag value; If true, samples generation stops, and final evaluation is calculated.
; - A vector; it will be taken as the input value on the next sample generation, i.e. iterated.
; - A vector; It is taken as the initial parameter for the following sample function:
; - A function; It is used for providing subsequent values to the NN. Described in detail later.
; - A vector; It is taken as the initial parameter for the following sample function.
; - A function; It is used to interpret subsequent NN outputs. Described in detail later.
; I.e. it needs to have the following definition pattern:
; (defn take-next-sample
;  [user-vector]
;  [flag updated-user-vector initial-provide-input-vector provide-input
;    initial-interpret-output-vector interpret-output])
(defn take-next-sample
"Generate the next sample to be evaluated by a NN"
  [user-vector]
  (let [sample-counter (get user-vector 0)
        m (int (rand 50)) n (int (rand 50)) mn (* m n)
       ; Numbers to factorize, and their multiply. A new random pair is generated every sample.
        max-iterations 1000]
       ; This is how many network iterations we allow until we consider it unstoppable.
  [(= 0 sample-counter)   ; The flag; if sample-counter runs down to 0, sampling stops.
    [(dec sample-counter)] ; The updated user vector; In this case it's the counter decreased.
    [mn] ; The initial vector to be supplied to the provide-input function; The number to factorize.
    (fn [state-vector user-vector] ; The provide-input function
        [(assoc state-vector 0 mn) ; Replace first number in the network state-vector with m*n
        [0.0]])  ; subsequently, zeros are provided, i.e. the input value is provided only once
    [max-iterations]; The initial vector for the interpret-output function; just a safeguard counter
    (fn [state-vector user-vector] ; The interpret-output function
        (let [iterations-left (get user-vector 0)
             [a b] (take-last 2 state-vector) ; Take the two last neurons' output as the NN output
             factorization-found (= (* (int a) (int b)) mn)] ; true if a*b = mn
           [(or (= iterations-left 0) factorization-found)
             [(dec iterations-left)]
             (if factorization-found 1.0 0.0)]))])) ; The partial evaluation; 1.0 if solution found.

; 'calculate-final-evaluation': this function takes a list of partial evaluations (any type),
; and returns a 'double' type numeric value, interpreted as the final network evaluation.
(defn calculate-final-evaluation
"Calculate the final evaluation from partials"
  [partial-evaluations]
  (reduce + partial-evaluations)) ; In our example we just sum the values up.

; Description for the provide-input function:
;- Creates a portion of input data for the next network iteration cycle,
;- Needs to be a valid Clojure function definition,
;- Needs to take and return a vector consisting of:
;  - a 'network-size' vector of 'double' values, and
;  - a user-defined vector, intended to keep own state.
; Initially, the initial-provide-input-vector is taken as the user-vector.

; Description for the interpret-output function:
;- Interprets a portion of output data,
;- Needs to be a valid Clojure function definition,
;- Needs to take a vector of
;  - a 'network-size' vector of 'double' values, and
;  - any user-defined vector, intended to keep own state,
;- Needs to return a vector of
;  - a boolean value interpreted as 'stop sample evaluation' (true => stop),
;  - a user-defined vector
;  - a double type value, taken as partial evaluation when the calculation is interrupted
; Initially, the initial-interpret-output-vector is taken as the user-vector.

;;; The mutable part
(def params
{
:population-save-interval '(10 "New file is saved and evals displayed every % generation; integer")
:population-save-folder '("factorization/" "New files are saved to this folder")
:mutation-probability-inverse '(12 "Self-adjustable. P of a weight modification when a new org is created; integer")
:crossover-probability '(0.4 "...when a new organism is created; double")
:initialize-population '(true  "If true, a new, zeroed population will be created; boolean")
:default-null-eval '(0.0 "Initial evaluation taken if initialize-population; double")
:iterations-per-input '(1 "Network cycles per each value provided; integer")
:population-size '(50 "The population size; integer")
:network-size '(12 "The network size; integer")
:generation '(0 "Current generation; integer")
:parallelism '(4 "# of organisms evaluated in parallel - if 0 then # of the JVM cores; integer")
})

; The population itself; made of pairs: evaluation, and an organism
(def population [])
