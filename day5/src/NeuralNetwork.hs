-- |= Neural Network Building Blocks
--
-- Note that some functions have been updated w.r.t massiv-0.4.3.0,
-- most notably changed Data.Massiv.Array.Numeric

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeuralNetwork
  ( ConvNet
  , Layer (..)
  , Matrix
  , Vector
  , FActivation (..)
  , sigmoid
  , sigmoid'
  , relu
  , relu'
  , genWeights
  , forward

  -- * Training
  , sgd

  -- * Inference
  , accuracy
  , avgAccuracy
  , winnerTakesAll

  -- * Helpers
  , rows
  , cols
  , computeMap
  , rand
  , randn
  , randomishArray
  , iterN
  , br
  ) where

import           Control.Monad ( replicateM, foldM )
import           Control.Applicative ( liftA2 )
import qualified System.Random as R
import           System.Random.MWC ( createSystemRandom )
import           System.Random.MWC.Distributions ( standard )
import           Data.List ( maximumBy )
import           Data.Ord
import           Data.Massiv.Array hiding ( map, zip, zipWith )
import qualified Data.Massiv.Array as A
import           Streamly
import qualified Streamly.Prelude as S

-- Note that images are volumes of channels x width x height, whereas
-- mini-batches are volumes-4 of batch size x channels x width x height.
-- Similarly, convolutional filter weights are volumes-4 of
-- out channels x in channels x kernel width x kernel height.
type Vector a = Array U Ix1 a
type Matrix a = Array U Ix2 a
-- type Volume a = Array U Ix3 a
type Volume4 a = Array U Ix4 a

-- Activation function symbols:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data FActivation = Relu | Sigmoid | Id

-- Neural network layers: Linear, Conv2d, Activation
data Layer a = Linear (Matrix a) (Vector a)
               | Conv2d (Volume4 a)
               | Activation FActivation

-- The main difference from the previous NeuralNetwork type
-- is that the network input is a volume, not a vector
-- (or volume-4, not matrix when in a batch)
type ConvNet a = [Layer a]

data Gradients a = -- Weight and bias gradients
                   LinearGradients (Matrix a) (Vector a)
                   -- Conv 2D gradients (no biases, but could be)
                   | Conv2dGradients (Volume4 a)
                   | NoGrad  -- No learnable parameters

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Lookup activation function by a symbol
getActivation :: FActivation -> (Matrix Float -> Matrix Float)
getActivation Id = id
getActivation Sigmoid = sigmoid
getActivation Relu = relu

-- | Lookup activation function derivative by a symbol
getActivation'
  :: FActivation
  -> (Matrix Float -> Matrix Float -> Matrix Float)
getActivation' Id = flip const
getActivation' Sigmoid = sigmoid'
getActivation' Relu = relu'

-- | Elementwise sigmoid computation
sigmoid :: Index ix => Array U ix Float -> Array U ix Float
sigmoid = computeMap f
  where
    f x = recip $ 1.0 + exp (-x)

-- | Compute sigmoid gradients
sigmoid' :: forall ix. Index ix
         => Array U ix Float
         -> Array U ix Float
         -> Array U ix Float
sigmoid' x dY =
  let sz = size x
      ones = A.replicate Par sz 1.0 :: Array D ix Float
      y = sigmoid x
      y' = delay y

      -- compute $ dY .*. y .*. (ones .-. y)
      e1 = maybe (error "Inconsistent dimensions#1") id (ones .-. y')
      e2 = maybe (error "Inconsistent dimensions#2") id (y' .*. e1)
      e3 = maybe (error "Inconsistent dimensions#3") compute (delay dY .*. e1)
  in e3

relu :: Index ix => Array U ix Float -> Array U ix Float
relu = computeMap (max 0.0)

relu' :: Index ix
      => Array U ix Float
      -> Array U ix Float
      -> Array U ix Float
relu' x = compute. A.zipWith f x
  where
    f x0 dy0 = if x0 <= 0
                  then 0
                  else dy0

randomishArray
  :: (Mutable r ix e, R.RandomGen a, R.Random e) =>
     (e, e) -> a -> Sz ix -> Array r ix e
randomishArray rng g0 sz = compute $ unfoldlS_ sz _rand g0
  where
    _rand g =
      let (a, g') = R.randomR rng g
      in (g', a)

-- | Uniformly-distributed random numbers Array
rand
  :: (R.Random e, Mutable r ix e) =>
     (e, e) -> Sz ix -> IO (Array r ix e)
rand rng sz = do
  g <- R.newStdGen
  return $ randomishArray rng g sz

-- | Random values from the Normal distribution
-- randn
--   :: (Fractional e, Index ix, Resize r Ix1, Mutable r Ix1 e)
--   => Sz ix -> IO (Array r ix e)
randn sz = do
    g <- createSystemRandom
    xs <- _nv g (totalElem sz)
    return $ resize' sz (fromList Seq xs)
  where
    _nv gen n = replicateM n (realToFrac <$> standard gen)
    {-# INLINE _nv #-}

rows :: Matrix Float -> Int
rows m =
  let (r :. _) = unSz $ size m
  in r

cols :: Matrix Float -> Int
cols m =
  let (_ :. c) = unSz $ size m
  in c

-- _scale = (*.)

-- scale :: Index sz => Float -> Array U sz Float -> Array U sz Float
-- scale konst = computeMap (* konst)

computeMap :: (Source r2 ix e', Mutable r1 ix e) =>
  (e' -> e) -> Array r2 ix e' -> Array r1 ix e
computeMap f = A.compute. A.map f

linearW' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearW' x dy =
  let trX = compute $ transpose x :: Matrix Float
      prod = maybe (error "Inconsistent dimensions in linearW'") id (trX |*| dy)
      m = recip $ fromIntegral (rows x)
  in compute $ m *. (delay prod)

linearX' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearX' w dy = maybe (error "Inconsistent dimensions in linearX'") compute (dy `multiplyTransposed` w)

-- | Bias gradient
bias' :: Matrix Float -> Vector Float
bias' dY = compute $ m *. (_sumRows $ delay dY)
  where
    m = recip $ fromIntegral $ rows dY

-- | Forward pass in a neural network:
-- exploit Haskell lazyness to never compute the
-- gradients.
forward
  :: ConvNet Float -> Volume4 Float -> Matrix Float
forward net dta = fst $ pass Eval net (dta, undefined)

softmax :: Matrix Float -> Matrix Float
softmax x =
  let x0 = expA (delay x)
      x1 = computeAs U $ _sumCols x0  -- Note _sumCols, not _sumRows
      x2 = x1 `colsLike` x
  in maybe (error  "Inconsistent dimensions in softmax") compute (x0 ./. x2)

-- | Both forward and backward neural network passes
pass
  :: Phase
  -- ^ `Train` or `Eval`
  -> ConvNet Float
  -- ^ `ConvNet` `Layer`s: weights and activations
  -> (Volume4 Float, Matrix Float)
  -- ^ Mini-batch with labels
  -> (Matrix Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass = undefined

-- | Broadcast a vector in Dim2
rowsLike :: Manifest r Ix1 Float
         => Array r Ix1 Float -> Matrix Float -> Array D Ix2 Float
rowsLike v m = br (Sz (rows m)) v

-- | Broadcast a vector in Dim1
colsLike :: Manifest r Ix1 Float
         => Array r Ix1 Float -> Matrix Float -> Array D Ix2 Float
colsLike v m = br1 (Sz (cols m)) v

-- | Broadcast by the given number of rows
br :: Manifest r Ix1 Float
   => Sz1 -> Array r Ix1 Float -> Array D Ix2 Float
br rows' v = expandWithin Dim2 rows' const v

-- | Broadcast by the given number of cols
br1 :: Manifest r Ix1 Float
   => Sz1 -> Array r Ix1 Float -> Array D Ix2 Float
br1 rows' v = expandWithin Dim1 rows' const v

-- | Stochastic gradient descent
sgd :: Monad m
  => Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> ConvNet Float
  -- ^ Neural network
  -> SerialT m (Volume4 Float, Matrix Float)
  -- ^ Data stream
  -> m (ConvNet Float)
sgd lr n net0 dataStream = iterN n epochStep net0
  where
    epochStep net = S.foldl' g net dataStream

    g :: ConvNet Float
      -> (Volume4 Float, Matrix Float)
      -> ConvNet Float
    g net dta =
      let (_, dW) = pass Train net dta
      in (zipWith f net dW)

    f :: Layer Float -> Gradients Float -> Layer Float

    -- Update Linear layer weights
    f (Linear w b) (LinearGradients dW dB) =
      let w1 = subtractGradMaybe lr w dW
          b1 = subtractGradMaybe lr b dB
      in Linear w1 b1

    f (Conv2d w) (Conv2dGradients dW) = Conv2d (subtractGradMaybe lr w dW)

    -- No parameters to change
    f layer NoGrad = layer

    f _ _ = error "Layer/gradients mismatch"

subtractGrad
  :: (Num e, MonadThrow m, Source r1 ix e, Source r2 ix e) =>
     e -> Array r1 ix e -> Array r2 ix e -> m (Array D ix e)
subtractGrad lr w dW = delay w .-. (lr *. delay dW)

subtractGradMaybe
  :: (Mutable r ix e, Num e, Source r1 ix e, Source r2 ix e) =>
     e -> Array r1 ix e -> Array r2 ix e -> Array r ix e
subtractGradMaybe lr w dW = maybe (error "Inconsistent dimensions") compute (subtractGrad lr w dW)

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1..n]

-- | Generate random weights and biases
genWeights
  :: (Int, Int)
  -> IO (Matrix Float, Vector Float)
genWeights (nin, nout) = do
  w <- setComp Par <$> _genWeights (nin, nout)
  b <- setComp Par <$> _genBiases nout
  return (w, b)
    where
      _genWeights (nin', nout') = do
          a <- randn sz :: IO (Matrix Float)
          return (compute $ k *. (delay a))
        where
          sz = Sz (nin' :. nout')
          k = 0.01

      _genBiases n = randn (Sz n)

maxIndex :: (Ord a, Num b, Enum b) => [a] -> b
maxIndex xs = snd $ maximumBy (comparing fst) (zip xs [0..])

winnerTakesAll ::
  Matrix Float  -- ^ Mini-batch of vectors
  -> [Int]  -- ^ List of maximal indices
winnerTakesAll m = map maxIndex xs
  where
    xs = toLists2 m

errors :: Eq lab => [(lab, lab)] -> [(lab, lab)]
errors = filter (uncurry (/=))
{-# SPECIALIZE errors :: [(Int, Int)] -> [(Int, Int)] #-}

accuracy :: (Eq a, Fractional acc) => [a] -> [a] -> acc
accuracy tgt pr = 100 * r
  where
    errNo = length $ errors (zip tgt pr)
    r = 1 - fromIntegral errNo / fromIntegral (length tgt)
{-# SPECIALIZE accuracy :: [Int] -> [Int] -> Float #-}

_accuracy :: ConvNet Float
  -> (Volume4 Float, Matrix Float)
  -> Float
-- NB: better avoid double conversion to and from one-hot-encoding
_accuracy net (batch, labelsOneHot) =
  let batchResults = winnerTakesAll $ forward net batch
      expected = winnerTakesAll labelsOneHot
  in accuracy expected batchResults

avgAccuracy
  :: Monad m
  => ConvNet Float
  -> SerialT m (Volume4 Float, Matrix Float)
  -> m Float
avgAccuracy net stream = s // len
  where
    results = S.map (_accuracy net) stream
    s = S.sum results
    len = fromIntegral <$> S.length results
    (//) = liftA2 (/)

-- | Sum values in each column and produce a delayed 1D Array
_sumRows :: Array D Ix2 Float -> Array D Ix1 Float
_sumRows = A.foldlWithin Dim2 (+) 0.0

-- | Sum values in each row and produce a delayed 1D Array
_sumCols :: Array D Ix2 Float -> Array D Ix1 Float
_sumCols = A.foldlWithin Dim1 (+) 0.0

-- TODO: modify the demo so that only the forward pass is defined.
-- Then, use `backprop` package for automatic differentiation.
