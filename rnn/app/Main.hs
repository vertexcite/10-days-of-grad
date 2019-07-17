{-# LANGUAGE FlexibleContexts #-}

-- | = Recurrent network approximation as a deep network
--
-- In this demo we train a model able to generate a sine trajectory
-- over time. This obviously requires an internal state (memory) since
-- from any point except the extrema there are always two valid
-- paths: either up or down. Moreover there is a need to store
-- a part of trajectory to be able to tell how fast you go down (up).

import           Data.Massiv.Array hiding ( map, zip, unzip, replicate, mapM )
import qualified Data.Massiv.Array as A
import           Streamly
import qualified Streamly.Prelude as S
import           Text.Printf ( printf )

import           NeuralNetwork

-- | Generate a stream of training data:
sineStream
  :: Int
  -- ^ Number of training samples
  -> Int
  -- ^ Number of inputs per training sample
  -> IO (SerialT IO ([Vector Float], Vector Float))
  -- ^ A stream containing tuples of inputs and desired targets
sineStream n n_inputs = return $ S.fromList (take n ys)
  where
    singletons = map A.singleton
    xs = [0,0.02..]  -- An infinite list starting at 0 with a step of 0.02
    pts = map (\x -> sin (2 * pi * x)) xs :: [Float]
    chunks = window (n_inputs + 1) pts :: [[Float]]
    ys = map (\c -> let d = singletons c
                    in (init d, last d)) chunks

-- Used for verification
sine :: [Vector Float]
sine =
  let xs = [0.5,0.52..10]
  in map (\x -> A.singleton $ sin (2 * pi * x)) xs

-- | A sliding window
window :: Int -> [a] -> [[a]]
window _ [] = []
window sz xs = take sz xs: window sz (drop stride xs)
  where
    stride = 1

main :: IO ()
main = do
  let [i, h, o] = [1, 100, 1]
      -- Hidden layers to approximate the recurrent layer
      -- when "unrolling" the network
      hidden_layers = 2

  s <- sineStream 200 hidden_layers
  let epochs = 10
      lr = 0.1

  (wI, _) <- genWeights (i, h)
  (wX, bX) <- genWeights (h, h)
  (wR, _) <- genWeights (h, o)

  let net0 = (wI, wX, bX, wR)
      fAct = Relu
      -- fAct = Tanh

  net' <- sgdRNN hidden_layers lr epochs net0 fAct s

  print net'

  let initial = 50
      seedSeq = take initial sine

      -- Initialize RNN state x0
      (t0, x0) = initRNN net' fAct seedSeq

      -- Use own inputs
      pt = head $ drop initial sine  -- Initial input
      t = runRNN net' fAct x0 (pt: t)
      test = take (length sine - initial) t

  -- TODO: toFloat each singleton
  -- Prelude.mapM_ (putStrLn. printf "%.3f %.3f") $ zip sine (t0 ++ test)
  Prelude.mapM_ print $ zip sine (t0 ++ test)
