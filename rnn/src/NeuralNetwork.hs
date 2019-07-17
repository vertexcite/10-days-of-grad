-- |= Neural Network Building Blocks
--
-- The idea of this module is to manage gradients manually.
-- That is done intentionally to illustrate neural
-- networks training.

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeuralNetwork
  ( NeuralNetwork
  , Layer (..)
  , Matrix
  , Vector
  , FActivation (..)
  , sigmoid
  , sigmoid'
  , genWeights
  , forward

  -- * Training
  , sgd

  -- * Inference
  , accuracy
  , avgAccuracy
  , inferBinary
  , RNNState (..)
  , rnn
  , runRNN
  , runRNN0
  , winnerTakesAll

  -- * Helpers
  , rows
  , cols
  , computeMap
  , rand
  , randn
  , randomishArray
  , scale
  , iterN
  , mean
  , var
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

type MatrixPrim r a = Array r Ix2 a
type Matrix a = Array U Ix2 a
type Vector a = Array U Ix1 a

-- Activation function symbols:
-- * Rectified linear unit (ReLU)
-- * Sigmoid
-- * Identity (no activation)
data FActivation = Relu | Sigmoid | Id

-- Neural network layers
data Layer a = Linear (Matrix a) (Vector a)
               -- Same as Linear, but without biases
               | Linear' (Matrix a)
               | Activation FActivation

type NeuralNetwork a = [Layer a]

data Gradients a = -- Weight and bias gradients
                   LinearGradients (Matrix a) (Vector a)
                   -- Weight gradients
                   | Linear'Gradients (Matrix a)
                   | NoGrad  -- No learnable parameters

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Lookup activation function by a symbol
getActivation :: Index ix =>
  FActivation -> (Array U ix Float -> Array U ix Float)
getActivation Id = id
getActivation Sigmoid = sigmoid
getActivation Relu = relu

-- | Lookup activation function derivative by a symbol
getActivation'
  :: Index ix => FActivation
  -> (Array U ix Float -> Array U ix Float -> Array U ix Float)
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
      ones = A.replicate Par sz 1.0 :: Array U ix Float
      y = sigmoid x
  in compute $ dY .* y .* (ones .- y)

relu :: Index ix => Array U ix Float -> Array U ix Float
relu = computeMap f
  where
    f x = if x < 0
             then 0
             else x

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
randomishArray rng g0 sz = compute $ unfoldlS_ Seq sz _rand g0
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
randn
  :: (Fractional e, Index ix, Resize r Ix1, Mutable r Ix1 e)
  => Sz ix -> IO (Array r ix e)
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

-- Returns a delayed Array. Useful for fusion
_scale :: (Num e, Source r ix e) =>
  e -> Array r ix e -> Array D ix e
_scale c = A.map (* c)

scale :: Index sz => Float -> Array U sz Float -> Array U sz Float
scale konst = computeMap (* konst)

computeMap :: (Source r2 ix e', Mutable r1 ix e) =>
  (e' -> e) -> Array r2 ix e' -> Array r1 ix e
computeMap f = A.compute. A.map f

linearW' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearW' x dy =
  let trX = compute $ transpose x
      prod = trX |*| dy
      m = recip $ fromIntegral (rows x)
  in m `scale` prod

linearX' :: Matrix Float
        -> Matrix Float
        -> Matrix Float
linearX' w dy = compute $ dy `multiplyTransposed` w

-- | Bias gradient
bias' :: Matrix Float -> Vector Float
bias' dY = compute $ m `_scale` _sumRows dY
  where
    m = recip $ fromIntegral $ rows dY

-- This type is to avoid parameter mismatch
newtype RNNState = State { state :: Vector Float }

-- | Recurrent neural network defined by the set of equations
--
-- \( x_n = f(W^I u_n + W^X x_{n-1} + b^X), \)
--
-- \( y_n = W^R x_n, \)
--
-- where \(u_n\) is the \(n\)-th input and \(f(x)\) is an element-wise
-- nonlinear activation function. The network retains its internal
-- state \(x_n\) while receiving new inputs \(u_n\).
rnn
  :: (Matrix Float, Matrix Float, Vector Float, Matrix Float)
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> Vector Float  -- ^ Input \(u_n\)
  -> RNNState  -- ^ Previous hidden state \(x_{n-1}\)
  -> (Vector Float, RNNState)
  -- ^ Output \(y_n\) and new hidden state \(x_n\)
rnn (wI, wX, bX, wR) fAct dta (State x') = (flatten y, State (flatten x))
  where
    -- First, reshape input vectors to matrices since multiplication
    -- operator |*| is defined only for matrices.
    x1 = vec2m x'
    u = vec2m dta
    b = vec2m bX

    f = getActivation fAct
    x = f (compute ((wI |*| u) .+ (wX |*| x1) .+ b))
    y = wR |*| x

-- | Run a recurrent neural network from given initial hidden state.
runRNN
  :: (Matrix Float, Matrix Float, Vector Float, Matrix Float)
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> RNNState  -- ^ Initial hidden state \(x_{n-1}\)
  -> [Vector Float]  -- ^ Input vectors list
  -> [Vector Float]  -- ^ Outputs
runRNN w fAct x0 inps = map fst p
  where
    p = scanr q (undefined, x0) inps
    q inp (_, xprev) = rnn w fAct inp xprev

-- | Run a recurrent neural network from zero initial hidden state.
-- Inputs are given as a (lazy) list so that the network should be
-- able to provide its own outputs as its new inputs.
-- For instance:
--
-- >>> let results = runRNN0 w Sigmoid (dta: results)
-- >>> take 5 results
runRNN0
  :: (Matrix Float, Matrix Float, Vector Float, Matrix Float)
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> [Vector Float]  -- ^ Input vectors
  -> [Vector Float]  -- ^ Output vectors
runRNN0 w@(_, wX, _, _) fAct = runRNN w fAct (State x0)
  where
    x0 = A.replicate Par (Sz (rows wX)) 0 :: Vector Float

-- | Convert vector to an n×1 matrix
vec2m :: Vector Float -> Matrix Float
vec2m v = resize' sz v
  where
    sz = Sz (elemsCount v :. 1)

-- | Forward pass in a neural network:
-- exploit Haskell lazyness to never compute the
-- gradients.
forward
  :: NeuralNetwork Float -> Matrix Float -> Matrix Float
forward net dta = fst $ pass Eval net (dta, undefined)

softmax :: Matrix Float -> Matrix Float
softmax x =
  let x0 = compute $ expA x :: Matrix Float
      x1 = compute $ _sumCols x0 :: Vector Float  -- Sumcols in this case!
      x2 = x1 `colsLike` x
  in (compute $ x0 ./ x2)

-- | Both forward and backward neural network passes
pass
  :: Phase
  -- ^ `Train` or `Eval`
  -> NeuralNetwork Float
  -- ^ `NeuralNetwork` `Layer`s: weights and activations
  -> (Matrix Float, Matrix Float)
  -- ^ Mini-batch with labels
  -> (Matrix Float, [Gradients Float])
  -- ^ NN computation from forward pass and weights gradients
pass phase net (x, tgt) = (pred, grads)
  where
    (_, pred, grads) = _pass x net

    -- Computes a tuple of:
    -- 1) Gradients for further backward pass
    -- 2) NN prediction
    -- 3) Gradients of learnable parameters (where applicable)
    _pass inp [] = (loss', pred, [])
      where
        -- TODO: Make softmax/loss/loss gradient a part of SGD/Adam?
        pred = softmax inp

        -- Gradient of cross-entropy loss
        -- after softmax activation.
        loss' = compute $ pred .- tgt

    _pass inp (Linear w b:layers) = (dX, pred, LinearGradients dW dB:t)
      where
        -- Forward
        lin = compute $ (inp |*| w) .+ (b `rowsLike` inp)

        (dZ, pred, t) = _pass lin layers

        -- Backward
        dW = linearW' inp dZ
        dB = bias' dZ
        dX = linearX' w dZ

    _pass inp (Linear' w:layers) = (dX, pred, Linear'Gradients dW:t)
      where
        -- Forward
        lin = compute (inp |*| w)

        (dZ, pred, t) = _pass lin layers

        -- Backward
        dW = linearW' inp dZ
        dX = linearX' w dZ

    _pass inp (Activation symbol:layers) = (dY, pred, NoGrad:t)
      where
        y = getActivation symbol inp  -- Forward

        (dZ, pred, t) = _pass y layers

        dY = getActivation' symbol inp dZ  -- Backward

-- | Broadcast a vector in Dim2
rowsLike :: Manifest r Ix1 Float
      => Array r Ix1 Float -> Matrix Float -> MatrixPrim D Float
rowsLike v m = br (rows m) v

-- | Broadcast a vector in Dim1
colsLike :: Manifest r Ix1 Float
      => Array r Ix1 Float -> Matrix Float -> MatrixPrim D Float
colsLike v m = br1 (cols m) v

-- | Broadcast by the given number of rows
br :: Manifest r Ix1 Float
   => Int -> Array r Ix1 Float -> MatrixPrim D Float
br rows' = expandWithin Dim2 rows' const

-- | Broadcast by the given number of cols
br1 :: Manifest r Ix1 Float
   => Int -> Array r Ix1 Float -> MatrixPrim D Float
br1 rows' = expandWithin Dim1 rows' const

-- | Stochastic gradient descent
sgd :: Monad m
  => Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> NeuralNetwork Float
  -- ^ Neural network
  -> SerialT m (Matrix Float, Matrix Float)
  -- ^ Data stream
  -> m (NeuralNetwork Float)
sgd lr n net0 dataStream = iterN n epochStep net0
  where
    epochStep net = S.foldl' g net dataStream

    g :: NeuralNetwork Float
      -> (Matrix Float, Matrix Float)
      -> NeuralNetwork Float
    g net dta =
      let (_, dW) = pass Train net dta
      in zipWith f net dW

    f :: Layer Float -> Gradients Float -> Layer Float

    -- Update Linear layer weights
    f (Linear w b) (LinearGradients dW dB) =
      Linear (compute $ w .- lr `_scale` dW) (compute $ b .- lr `_scale` dB)

    f (Linear' w) (Linear'Gradients dW) =
      Linear' (compute $ w .- lr `_scale` dW)

    -- No parameters to change
    f layer NoGrad = layer

    f _ _ = error "Layer/gradients mismatch"

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1..n]

addC :: (Num e, Source r ix e) => Array r ix e -> e -> Array D ix e
addC m c = A.map (c +) m

-- | Generate random weights and biases
genWeights
  :: (Int, Int)
  -> IO (Matrix Float, Vector Float)
genWeights (nin, nout) = do
  w <- setComp Par <$> _genWeights (nin, nout)
  b <- setComp Par <$> _genBiases nout
  return (w, b)
    where
      _genWeights (nin', nout') = scale k <$> randn sz
        where
          sz = Sz (nin' :. nout')
          k = 0.01

      _genBiases n = randn (Sz n)

-- | Perform a binary classification
inferBinary
  :: NeuralNetwork Float -> Matrix Float -> Matrix Float
inferBinary net dta =
  let prediction = forward net dta
  -- Thresholding the NN output
  in compute $ A.map (\a -> if a < 0.5 then 0 else 1) prediction

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

_accuracy :: NeuralNetwork Float
  -> (Matrix Float, Matrix Float)
  -> Float
-- NB: better avoid double conversion to and from one-hot-encoding
_accuracy net (batch, labelsOneHot) =
  let batchResults = winnerTakesAll $ forward net batch
      expected = winnerTakesAll labelsOneHot
  in accuracy expected batchResults

avgAccuracy
  :: Monad m
  => NeuralNetwork Float
  -> SerialT m (Matrix Float, Matrix Float)
  -> m Float
avgAccuracy net stream = s // len
  where
    results = S.map (_accuracy net) stream
    s = S.sum results
    len = fromIntegral <$> S.length results
    (//) = liftA2 (/)

-- | Average elements in each column
mean :: Matrix Float -> Vector Float
mean ar = compute $ m `_scale` _sumRows ar
  where
    m = recip $ fromIntegral (rows ar)

-- | Variance over each column
var :: Matrix Float -> Vector Float
var ar = compute $ m `_scale` r
  where
    mu = br nRows $ mean ar
    nRows = rows ar
    r0 = compute $ (ar .- mu) .^ 2
    r = _sumRows r0
    m = recip $ fromIntegral nRows

-- | Sum values in each column and produce a delayed 1D Array
_sumRows :: Matrix Float -> Array D Ix1 Float
_sumRows = A.foldlWithin Dim2 (+) 0.0

-- | Sum values in each row and produce a delayed 1D Array
_sumCols :: Matrix Float -> Array D Ix1 Float
_sumCols = A.foldlWithin Dim1 (+) 0.0

-- TODO: another demo where only the forward pass is defined.
-- Then, use `backprop` package for automatic differentiation.
