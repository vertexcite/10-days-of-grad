{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE TypeOperators #-}

module NeuralNetwork
  ( NeuralNetwork (..)
  , Layer (..)
  , Activation (..)
  , genWeights
  , genNetwork

  -- * Training
  , optimize
--   , AdamParameters (..)
--   , adamParams
--   , optimizeAdam
--
--   -- * Inference
--   , inferBinary
--   , accuracy
  ) where

import Data.Array.Accelerate
  (
    Exp
  , Elt
  , Array
  , Matrix
  , Vector
  , Scalar
  , Z(..)
  , (:.)(..)
  , use
  )
import qualified Data.Array.Accelerate as A
-- GPU backend
import Data.Array.Accelerate.LLVM.PTX as PTX
import Data.Array.Accelerate.Numeric.LinearAlgebra
  (
    Numeric
  , (<>)
  )
import qualified Data.Array.Accelerate.System.Random.MWC as MWC

-- Activation function:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data Activation = Sigmoid | Tanh | Id

-- Neural network layer: weights, biases, and activation
data Layer a = Layer (Matrix a) (Matrix a) Activation

type NeuralNetwork a = [Layer a]

-- | Weight and bias gradients
data Gradients a = Gradients (Matrix a) (Matrix a)

-- | Lookup activation function by a symbol
getActivation :: Activation -> (Acc (Matrix Double) -> Acc (Matrix Double))
getActivation Id = id
getActivation Sigmoid = sigmoid
getActivation Tanh = NeuralNetwork.tanh

-- | Sigmoid activation function (vectorized)
sigmoid :: (A.Shape sh, Elt e, Floating (Exp e))
        => Acc (Array sh e)
        -> Acc (Array sh e)
sigmoid = A.map _sigmoid
  where
    _sigmoid x = recip $ 1.0 + exp (-x)

-- | Sigmoid gradient
sigmoid' :: (A.Shape sh, Elt e, Floating (Exp e))
         => Acc (Array sh e)
         -> Acc (Array sh e)
         -> Acc (Array sh e)
sigmoid' x dY = dY * y * (ones - y)
   where
    y = sigmoid x
    sh = A.shape y
    ones = A.fill sh 1.0

-- | Linear layer
linear :: Numeric e => Acc (Matrix e) -> Acc (Matrix e) -> Acc (Matrix e)
linear = (<>)

-- | Tanh activation function (vectorized)
tanh :: (A.Shape sh, Elt e, Floating (Exp e))
     => Acc (Array sh e)
     -> Acc (Array sh e)
tanh = A.map Prelude.tanh

-- | Lookup activation function derivative by a symbol
getActivation'
  :: Activation
  -> (Acc (Matrix Double) -> Acc (Matrix Double) -> Acc (Matrix Double))
-- getActivation' Id = flip const
getActivation' Sigmoid = sigmoid'
-- getActivation' Tanh = \x dY -> tanhGradient x * dY

-- -- | Forward pass in a neural network:
-- -- exploit Haskell lazyness to never compute the
-- -- gradients.
-- forward
--   :: NeuralNetwork Double -> Matrix Double -> Matrix Double
-- forward net dta = fst $ pass net (dta, undefined)
--
-- -- | Both forward and backward neural network passes
-- pass
--   :: NeuralNetwork Double
--   -- ^ `NeuralNetwork` `Layer`s: weights and activations
--   -> (Matrix Double, Matrix Double)
--   -- ^ Data set
--   -> (Matrix Double, [Gradients Double])
--   -- ^ NN computation from forward pass and weights gradients
-- pass net (x, tgt) = (pred, grads)
--   where
--     (_, pred, grads) = _pass x net
--
--     _pass inp [] = (loss', pred, [])
--       where
--         pred = sigmoid inp
--         -- Gradient of cross-entropy loss
--         -- after sigmoid activation.
--         loss' = pred - tgt
--
--     _pass inp (Layer w b sact:layers) = (dX, pred, Gradients dW dB:t)
--       where
--         lin = (inp LA.<> w) + b
--         y = getActivation sact lin
--
--         (dZ, pred, t) = _pass y layers
--
--         dY = getActivation' sact lin dZ
--         dW = linearW' inp dY
--         dB = bias' dY
--         dX = linearX' w dY
--
-- | Bias gradient
bias' :: (Numeric e, Fractional (Exp e), A.FromIntegral Int e)
      => Acc (Matrix e)
      -> Acc (Matrix e)
bias' dY = m `scale` r
  where
    -- Sum elements in each column
    v0 = A.sum (A.transpose dY)
    cols' = cols dY
    r = _unflatten1 cols' v0
    -- Scale factor
    m = recip $ A.fromIntegral $ rows dY

-- Reshape a `Vector` into a `Matrix` with singleton columns
_unflatten1 :: Elt e
            => Exp Int  -- ^ Number of columns
            -> Acc (Vector e)
            -> Acc (Matrix e)
_unflatten1 cols' = A.reshape (A.lift (Z :. (1 :: Exp Int) :. cols'))

-- | Linear layer weights gradient
linearW' :: (Numeric e, Fractional (Exp e), A.FromIntegral Int e)
        => Acc (Matrix e)
        -> Acc (Matrix e)
        -> Acc (Matrix e)
linearW' x dY = m `scale` (A.transpose x <> dY)
  where
    m = recip $ A.fromIntegral (rows x)

-- | Linear layer inputs gradient
linearX' :: (Numeric e, Fractional (Exp e))
        => Acc (Matrix e)
        -> Acc (Matrix e)
        -> Acc (Matrix e)
linearX' w dY = dY <> A.transpose w

optimize
  :: (A.Elt b, A.Shape sh, Num (Exp b))
    => Exp b
    -- ^ Learning rate
    -> Int
    -- ^ No of iterations
    -> (Acc (Array sh b) -> Acc (Array sh b))
    -> Acc (Array sh b)
    -> [Acc (Array sh b)]
optimize gamma iterN gradF x0 = take iterN (iterate step x0)
  where
    step x = x - (gamma `scale` (gradF x))

scale
  :: (Elt b, A.Shape sh, Num (Exp b)) =>
     Exp b -> Acc (Array sh b) -> Acc (Array sh b)
scale konst = A.map (konst *)

instance (A.Shape sh, Elt e, Num (Exp e)) => Num (Acc (Array sh e))
  where
    (+) = A.zipWith (+)
    (-) = A.zipWith (-)
    (*) = A.zipWith (*)
    negate = A.map Prelude.negate
    signum = A.map Prelude.signum
    fromInteger = undefined
    -- Element-wise absolute value.
    -- For instance, proper absolute value of a Vector is
    -- `unit (sqrt(fold (+) $ A.zipWith (^2)))`.
    -- Note, there exists also a more precise `sum` implementation
    -- in Data.Array.Accelerate.Numeric.Sum.
    abs = A.map Prelude.abs

-- -- | Gradient descent optimization
-- optimize
--   :: Double
--   -- ^ Learning rate
--   -> Int
--   -- ^ No of iterations
--   -> NeuralNetwork Double
--   -- ^ Neural network
--   -> (Matrix Double, Matrix Double)
--   -- ^ Dataset
--   -> NeuralNetwork Double
--   -- ^ Updated neural network
-- optimize lr iterN net0 dataSet = last $ take iterN (iterate step net0)
--   where
--     step net = zipWith f net dW
--       where
--         (_, dW) = pass net dataSet
--
--     f :: Layer Double
--       -> Gradients Double
--       -> Layer Double
--     f (Layer w b act) (Gradients dW dB) =
--       Layer (w - lr `scale` dW) (b - lr `scale` dB) act
--
-- data AdamParameters = AdamParameters { _beta1 :: Double
--                                      , _beta2 :: Double
--                                      , _epsilon :: Double
--                                      , _lr :: Double
--                                      }
--
-- -- | Adam optimizer parameters
-- adamParams = AdamParameters { _beta1 = 0.9
--                             , _beta2 = 0.999
--                             , _epsilon = 1e-8
--                             , _lr = 0.001  -- ^ Learning rate
--                             }
--
-- -- | Adam optimization
-- optimizeAdam
--   :: AdamParameters
--      -- ^ Adam parameters
--      -> Int
--      -- ^ No of iterations
--      -> NeuralNetwork Double
--      -- ^ Neural network layers
--      -> (Matrix Double, Matrix Double)
--      -- ^ Dataset
--      -> NeuralNetwork Double
-- optimizeAdam p iterN w0 dataSet = w
--   where
--     s0 = map zf w0
--     v0 = map zf w0
--     zf (Layer a b _) = (zerosLike a, zerosLike b)
--     zerosLike m = matrix c (replicate (r*c) 0.0)
--       where
--         r = rows m
--         c = cols m
--     (w, _, _) = _adam p iterN (w0, s0, v0) dataSet
--
-- _adam
--   :: AdamParameters
--      -> Int
--      -> ([Layer Double], [(Matrix Double, Matrix Double)], [(Matrix Double, Matrix Double)])
--      -> (Matrix Double, Matrix Double)
--      -> ([Layer Double], [(Matrix Double, Matrix Double)], [(Matrix Double, Matrix Double)])
-- _adam p@AdamParameters { _lr = lr
--                        , _beta1 = beta1
--                        , _beta2 = beta2
--                        , _epsilon = epsilon
--       } iterN (w0, s0, v0) dataSet = last $ take iterN (iterate step (w0, s0, v0))
--   where
--     step (w, s, v) = (wN, sN, vN)
--       where
--         (_, dW) = pass w dataSet
--
--         sN = zipWith f2 s dW
--         vN = zipWith f3 v dW
--         wN = zipWith3 f w vN sN
--
--         f :: Layer Double
--           -> (Matrix Double, Matrix Double)
--           -> (Matrix Double, Matrix Double)
--           -> Layer Double
--         f (Layer w_ b_ sf) (vW, vB) (sW, sB) =
--            Layer (w_ - lr `scale` vW / ((sqrt sW) `addC` epsilon))
--                  (b_ - lr `scale` vB / ((sqrt sB) `addC` epsilon))
--                  sf
--
--         addC m c = cmap (+ c) m
--
--         f2 :: (Matrix Double, Matrix Double)
--            -> Gradients Double
--            -> (Matrix Double, Matrix Double)
--         f2 (sW, sB) (Gradients dW dB) =
--           ( beta2 `scale` sW + (1 - beta2) `scale` (dW^2)
--           , beta2 `scale` sB + (1 - beta2) `scale` (dB^2))
--
--         f3 :: (Matrix Double, Matrix Double)
--            -> Gradients Double
--            -> (Matrix Double, Matrix Double)
--         f3 (vW, vB) (Gradients dW dB) =
--           ( beta1 `scale` vW + (1 - beta1) `scale` dW
--           , beta1 `scale` vB + (1 - beta1) `scale` dB)
--
-- -- | Perform a binary classification
-- inferBinary
--   :: NeuralNetwork Double -> Matrix Double -> Matrix Double
-- inferBinary net dta =
--   let pred = forward net dta
--   -- Thresholding the NN output
--   in cmap (\a -> if a < 0.5 then 0 else 1) pred

-- | Generate random weights and biases
genWeights
  :: (Elt e, MWC.Variate e, Floating e)
  => (Int, Int)
  -> IO (Matrix e, Matrix e)
genWeights (nin, nout) = do
  w <- _genWeights (nin, nout)
  b <- _genWeights (1, nout)
  return (w, b)
    where
      _genWeights (nin, nout) = do
          let sh = Z :. nin :. nout
              k = recip $ sqrt (1.0 / fromIntegral nin)
          MWC.randomArray (MWC.uniformR (-k, k)) sh

-- | Generate a neural network with random weights
genNetwork
  :: [Int] -> [Activation] -> IO (NeuralNetwork Double)
genNetwork nodes activations = do
    weights <- mapM genWeights nodes'
    return (zipWith (\(w, b) a -> Layer w b a) weights activations)
  where
    nodes' = zip nodes (tail nodes)

-- -- | Binary classification accuracy in percent
-- accuracy
--   :: [Layer Double]
--   -- ^ Neural network
--   -> (Matrix Double, Matrix Double)
--   -- ^ Dataset
--   -> Double
-- accuracy net (dta, tgt) = 100 * (1 - e / m)
--   where
--     pred = net `inferBinary` dta
--     e = sumElements $ abs (tgt - pred)
--     m = fromIntegral $ rows tgt

-- | Number of matrix rows
rows :: Elt e => Acc (Matrix e) -> Exp Int
rows matr = rows'
  where
    Z :. rows' :. _ = A.unlift (A.shape matr) :: Z :. Exp Int :. Exp Int

-- | Number of matrix columns
cols :: Elt e => Acc (Matrix e) -> Exp Int
cols matr = cols'
  where
    Z :. _ :. cols' = A.unlift (A.shape matr) :: Z :. Exp Int :. Exp Int
