(ns clj-djl-examples.predict
  (:require [clj-djl.modality.cv :as cv]
            [clj-djl.model :as model]
            [clj-djl.ndarray :as ndarray]
            [clj-djl.model-zoo :as zoo]))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (let [img (cv/download-image-from "https://djl-ai.s3.amazonaws.com/resources/images/0.png")
        model
        (-> (model/new-instance "mlp")
            (model/set-block (zoo/mlp (* 28 28) 10 (int-array 128 64)))
            (model/load "build/mlp"))
        translator
        (-> (ai.djl.modality.cv.translator.ImageClassificationTranslator/builder)
            (.addTransform (ai.djl.modality.cv.transform.ToTensor.))
            (.optSynset (->> (range 0 10)
                             (map str)))
            (.build))
        predicator (model/new-predictor
                    model
                    translator)]
    (println (str (model/predict predicator img)))))
