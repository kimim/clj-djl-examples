(defproject clj-djl-examples "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [clj-djl "0.1.5"]
                 [clj-djl/dataframe "0.1.2"]]
  :main ^:skip-aot clj-djl-examples.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
