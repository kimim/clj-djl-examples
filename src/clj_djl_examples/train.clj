(ns clj-djl-examples.train
  (:gen-class)
  (:require [clojure.java.io :as io]
            [clj-djl.ndarray :as ndarray]
            [clj-djl.model :as model]
            [clj-djl.model-zoo :as zoo]
            [clj-djl.training :as train]
            [clj-djl.engine :as engine]
            [clj-djl.training.dataset :as ds])
  (:import [ai.djl.basicdataset Mnist]
           [java.nio.file Paths]))

(defn train-mnist []
  (let [mnist   ;;dataset
        (-> (Mnist/builder)
            (ds/set-sampling 32 true)
            (ds/build)
            (ds/prepare (train/new-progress-bar)))
        model   ;;model from model zoo
        (-> (model/new-instance "mlp")
            (model/set-block (zoo/mlp (* 28 28) 10 (int-array 128 64))))
        config  ;;trainer configs
        (-> (train/softmax-cross-entropy-loss)
            (train/new-training-config)
            (train/add-evaluator (train/new-accuracy))
            (train/add-training-listeners (train/new-default-training-listeners)))
        trainer  ;;trainer
        (-> (train/new-trainer model config)
            (train/initialize [(ndarray/shape 1 (* 28 28))]))]
    (doseq [epoch (range 3)] ;;train for 3 epoch
      (doseq [batch (train/iterate-dataset trainer mnist)]
        (train/train-batch trainer batch)
        (train/step trainer)
        (train/close batch))
      (train/notify-listeners trainer (fn [listner] (.onEpoch listner trainer))))
    model))


(defn save-model [model path]
  (let [nio-path (java.nio.file.Paths/get path (into-array [""]))]
    (io/make-parents path)
    (model/set-property model "Epoch" "3")
    (model/save model nio-path "mlp")))


(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (-> (train-mnist)
      (save-model "build/mlp"))
  (println "model trained and saved for mnist dataset!"))
