{-# LANGUAGE FlexibleContexts #-}

-- | = Recurrent network approximation as a deep network
--
-- In this demo we train a model able to generate a sine trajectory
-- over time. This obviously requires an internal state (memory) since
-- from any point except the extrema there are always two valid
-- paths: either up or down.

import           Data.Massiv.Array hiding ( map, zip, unzip, replicate, mapM )
import qualified Data.Massiv.Array as A
import           Streamly
import qualified Streamly.Prelude as S
import           Text.Printf ( printf )

import           NeuralNetwork

main :: IO ()
main = do
  let [i, h, o] = [1, 100, 1]
      -- Hidden layers to approximate the recurrent layer
      -- when "unrolling" the network
      hidden_layers = 3

  -- Below, we generate weights for any number of hidden layers.
  -- An example of a recurrent network approximation having two
  -- hidden layers:
  --
  -- (wi, bi) <- genWeights (i, h)
  -- (wx, bx) <- genWeights (h, h)
  -- (wr, br) <- genWeights (h, o)
  -- let net = [ Linear wi bi
  --           , Activation Relu
  --
  --           -- First hidden layer
  --           , Linear wx bx
  --           , Activation Relu
  --
  --           -- Second hidden layer
  --           , Linear wx bx
  --           , Activation Relu
  --
  --           , Linear wr br
  --           ]
  --
  let hs = replicate hidden_layers h
  weights <- mapM genWeights $ zip (i: hs) hs

  -- Here is a fully-connected layer with an activation
  let block (w, b) = [ Linear w b
                     , Activation Relu ]

  (wr, br) <- genWeights (h, o)
  let net = concatMap block weights ++ [Linear wr br]
  --
  -- The readout layer has an identity activation, so we simply
  -- put `Linear` as the final layer.
  --
  -- Later, we are going to modify the training procedure to
  -- simultaneously update the weights in the unrolled recurrent layer.

  -- -- Crucial parameters: initial weights magnitude and
  -- -- learning rate (lr)
  -- let epochs = 10
  --     lr = 0.1
  --
  -- net' <- sgd' lr epochs net trainS
  --
  -- putStrLn $ printf "%d training epochs" epochs

  return ()
