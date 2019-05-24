{-# LANGUAGE FlexibleContexts #-}

-- | = Batch normalization demo

import           Data.Massiv.Array as A
import           Streamly
import qualified Streamly.Prelude as S
import qualified System.Random as R
import           Text.Printf ( printf )

import           NeuralNetwork

-- | Circles dataset
makeCircles
  :: Int -> Float -> Float -> Int -> IO (Matrix Float, Matrix Float)
makeCircles m factor noiseLevel seed = do
  let rand' g n = randomishArray (0, 2 * pi) g (Sz (n :. 1))
      m1 = m `div` 2
      m2 = m - (m `div` 2)

  let g1 = R.mkStdGen seed
      (g2, g3) = R.split g1
  let r1 = rand' g1 m1 :: Matrix Float
      r2 = rand' g2 m2 :: Matrix Float
      ns = randomishArray (-noiseLevel / 2, noiseLevel / 2) g3 (Sz (m :. 2)) :: Matrix Float
  let outerX = compute $ cosA r1 :: Matrix Float
      outerY = compute $ sinA r1 :: Matrix Float
      innerX = scale factor $ compute $ cosA r2 :: Matrix Float
      innerY = scale factor $ compute $ sinA r2 :: Matrix Float

  -- Merge them all
  let x1 = compute $ append' 1 outerX outerY :: Matrix Float
      x2 = compute $ append' 1 innerX innerY :: Matrix Float
      x = compute $ append' 2 x1 x2 :: Matrix Float
      y1 = (A.replicate Par (Sz2 m1 1) 0) :: Matrix Float
      y2 = (A.replicate Par (Sz2 m2 1) 1) :: Matrix Float
      y = append' 2 y1 y2

  return (compute $ x .+ ns, compute y)

-- | Generate a stream of circles mini-datasets (mini-batches)
makeCirclesStream
  :: IsStream t =>
    [Int] -> Int -> Float -> Float -> t IO (Matrix Float, Matrix Float)
makeCirclesStream seeds batchSize factor noiseLevel =
  let gs = S.fromList seeds
  in S.mapM (makeCircles batchSize factor noiseLevel) gs

main :: IO ()
main = do
  -- By seeding, we ensure the stream is constant between epochs
  let trainStream = makeCirclesStream [1..6] 32 0.6 0.1
      testStream = makeCirclesStream [7..9] 32 0.6 0.1

  let [i, h1, o] = [2, 128, 1]
  (w1, b1) <- genWeights (i, h1)
  let ones = A.replicate Par (Sz1 h1) 1 :: Vector Float
      zeros = A.replicate Par (Sz1 h1) 0 :: Vector Float
  (w2, b2) <- genWeights (h1, o)
  let net = [ Linear w1 b1
            , Activation Relu
            , Batchnorm1d zeros ones ones zeros
            , Linear w2 b2
            ]
  -- No batchnorm layer
  let net2 = [ Linear w1 b1
             , Activation Relu
             , Linear w2 b2
             ]

  let epochs = 200
      lr = 0.001  -- Learning rate
  net' <- sgd lr epochs net trainStream
  net2' <- sgd lr epochs net2 trainStream

  putStrLn $ printf "Circles problem, 1 hidden layer of 128 neurons, %d epochs" epochs
  putStrLn "---"

  tacc <- net' `avgAccuracy` trainStream
  putStrLn $ printf "Training accuracy (SGD + batchnorm) %.1f" tacc

  acc <- net' `avgAccuracy` testStream
  putStrLn $ printf "Validation accuracy (SGD + batchnorm) %.1f" acc

  tacc <- net2' `avgAccuracy` trainStream
  putStrLn $ printf "Training accuracy (SGD) %.1f" tacc

  acc <- net2' `avgAccuracy` testStream
  putStrLn $ printf "Validation accuracy (SGD) %.1f" acc
