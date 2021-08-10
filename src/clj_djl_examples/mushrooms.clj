(ns clj-d2l-examples.mushrooms
  (:require
   [clj-djl.nn :as nn]
   [clj-djl.model :as m]
   [clj-djl.ndarray :as nd]
   [clj-djl.training :as t]
   [clj-djl.utils :as utils]
   [clj-djl.training.loss :as loss]
   [clj-djl.training.optimizer :as optimizer]
   [clj-djl.training.tracker :as tracker]
   [clj-djl.training.listener :as listener]
   [clj-djl.training.initializer :as initializer]
   [clj-djl.training.dataset :as ds]
   [clj-djl.dataframe :as df]
   [clj-djl.dataframe.column-filters :as cf])
  (:import [ai.djl.training.listener EvaluatorTrainingListener EpochTrainingListener]))

(def mushroom-csv (df/->dataframe "data/mushrooms.csv" {:parser-fn :string}))

(def features (-> mushroom-csv
                  (df/drop-columns  ["class"])
                  (df/categorical->one-hot :all)))
(def labels (->  mushroom-csv
                 (df/select-columns ["class"])
                 (df/categorical->one-hot :all)))

(def ndm (nd/new-base-manager))
(def X (df/->ndarray ndm features))
(def y (df/->ndarray ndm labels))

(def dataset (-> (ds/array-dataset {:data (nd/to-type X :float32 false)
                                    :labels (nd/to-type y :float32 false)
                                    :batchsize 256
                                    :shuffle true})))

(let [[mushroom-train mushroom-test] (ds/random-split dataset 70 30)
      net (nn/sequential {:blocks [(nn/linear {:units 256})
                                   (nn/relu-block)
                                   (nn/linear {:units 128})
                                   (nn/sigmoid-block)
                                   (nn/dropout {:rate 0.2})
                                   (nn/linear {:units 2})]
                          :initializer (nn/normal-initializer)})
      config (t/config {:loss (loss/l2-loss)
                        ;;:listeners (listener/logging)
                        :listeners [(EvaluatorTrainingListener.) (EpochTrainingListener.)]
                        :evaluator (t/accuracy)
                        :optimizer (optimizer/sgd {:tracker (tracker/fixed 0.01)})})
      model (m/model {:name "linear"
                      :block net})
      trainer (m/trainer model config)]
  (t/initialize trainer [(nd/shape 256 117)])
  (t/set-metrics trainer (t/metrics))
  (t/fit trainer 10 mushroom-train mushroom-test)
  (println (t/get-result trainer)))
