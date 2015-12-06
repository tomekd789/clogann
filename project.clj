(defproject clogann "0.1.0-SNAPSHOT"
  :description "CloGANN: breeding Neural Networks by a Genetic Algorithm in Clojure"
  :url "https://github.com/tomekd789/clogann"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.7.0"]]
  :main ^:skip-aot clogann.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
