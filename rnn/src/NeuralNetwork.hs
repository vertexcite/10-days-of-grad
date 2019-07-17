-- |= Neural Network Building Blocks
--
-- The idea of this module is to manage gradients manually.
-- That is done intentionally to illustrate neural
-- networks training.

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}

module NeuralNetwork
  ( Matrix
  , Vector
  , FActivation (..)
  , sigmoid
  , sigmoid'
  , genWeights

  -- * Training
  , sgdRNN

  -- * Inference
  , accuracy
  , RNNState (..)
  , rnn
  , runRNN
  , runRNN'
  , initRNN

  -- * Helpers
  , (#>)
  , (<#)
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

loss :: Index ix => Array U ix Float -> Array U ix Float -> Float
loss y tgt =
  let diff = y .- tgt
  in A.sum $ A.map (^2) diff

loss' :: Index ix
      => Array U ix Float -> Array U ix Float -> Array U ix Float
loss' y tgt =
  let diff = y .- tgt
  in compute (A.map (* 2) diff)

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

-- | Matrix by vector multiplication: result is a column-vector
infixr 7 #>
(#>) :: (Num a, Unbox a) => Matrix a -> Vector a -> Vector a
matr #> v = flatten r
  where
    v1 = vec2m v
    r = matr |*| v1

-- | Vector by matrix multiplication: result is a row-vector
infixr 7 <#
(<#) :: (Num a, Unbox a) => Vector a -> Matrix a -> Vector a
v <# matr = flatten r
  where
    v1 = vec2m' v
    r = v1 |*| matr

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
-- Just for convenience
type RNNWeights = (Matrix Float, Matrix Float, Vector Float, Matrix Float)
type RNNGradients = RNNWeights

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
  :: RNNWeights
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> Vector Float  -- ^ Input \(u_n\)
  -> RNNState  -- ^ Previous hidden state \(x_{n-1}\)
  -> (Vector Float, RNNState)
  -- ^ Output \(y_n\) and new hidden state \(x_n\)
--
-- rnn (wI, wX, bX, wR) fAct u (State x0) = (y, State x)
--   where
--     f = getActivation fAct
--     x = f (compute (wI #> u .+ wX #> x0 .+ bX))
--     y = wR #> x
--
-- Transposed version
rnn (wI, wX, bX, wR) fAct u (State x0) = (y, State x)
  where
    f = getActivation fAct
    x = f (compute (u <# wI .+ x0 <# wX .+ bX))
    y = x <# wR

-- | Run a recurrent neural network from given initial hidden state.
--
-- Inputs are given as a (lazy) list so that the network should be
-- able to provide its own outputs as its new inputs.
-- For instance:
--
-- >>> let results = runRNN w Sigmoid x0 (dta: results)
-- >>> take 5 results
runRNN
  :: RNNWeights
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> RNNState  -- ^ Initial hidden state \(x_{n-1}\)
  -> [Vector Float]  -- ^ Input vectors list
  -> [Vector Float]  -- ^ Outputs
runRNN w fAct x0 inps = map fst p
  where
    p = runRNN' w fAct x0 inps

runRNN'
  :: RNNWeights
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> RNNState  -- ^ Initial hidden state \(x_{n-1}\)
  -> [Vector Float]  -- ^ Input vectors list
  -> [(Vector Float, RNNState)]  -- ^ Outputs with recurrent layer states
runRNN' w fAct x0 inps = p
  where
    p = scanr q (undefined, x0) inps
    q inp (_, xprev) = rnn w fAct inp xprev

-- | Run a recurrent neural network from zero initial state
initRNN
  :: RNNWeights
  -- ^ Input weights \(W^I\), internal state weights \(W^X\), biases \(b^X\), and readout weights \(W^R\)
  -> FActivation  -- ^ Activation function \(f\)
  -> [Vector Float]  -- ^ Input vectors
  -> ([Vector Float], RNNState)  -- ^ Output vectors and final RNN state
initRNN w@(_, wX, _, _) fAct inp = (map fst p, snd $ last p)
  where
    x0 = A.replicate Par (Sz (rows wX)) 0 :: Vector Float
    p = runRNN' w fAct (State x0) inp

-- | Convert vector to an n×1 matrix
vec2m :: Unbox a => Vector a -> Matrix a
vec2m v = resize' sz v
  where
    sz = Sz (elemsCount v :. 1)

-- | Convert vector to an 1×n matrix
vec2m' :: Unbox a => Vector a -> Matrix a
vec2m' v = resize' sz v
  where
    sz = Sz (1 :. elemsCount v)

softmax :: Matrix Float -> Matrix Float
softmax x =
  let x0 = compute $ expA x :: Matrix Float
      x1 = compute $ _sumCols x0 :: Vector Float  -- Sumcols in this case!
      x2 = x1 `colsLike` x
  in (compute $ x0 ./ x2)

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

-- The first difference of this method compared to normal SGD is that
-- the network receives as many inputs as there are
-- 'hidden' layers. We simplify our task and only compute
-- gradients w.r.t. the last (desired) output, therefore
-- ignoring any intermediate outputs y_i.
--
-- The second difference is that all hidden layers are updated
-- with the same gradient simultaneously.
--
-- Below is an illustration of the forward and backward passes:
--
--    <--- dU
-- u1 -------> [] -----> [] -----> y1
--            |  ^
--        x1  |  |
--            |  | dX1
--            \/ |
--
--   u2 -----> [] -----> [] -----> y2
--            |  ^
--        x2  |  |
--            |  | dX2
--            \/ |
--                <--- dX3  <--- dy3
--     u3 ---> [] -----> [] -----> y3
--                  x3
--
sgdRNN :: Monad m
  => Int
  -- ^ Hidden layers for the recurrent layer approximation
  -> Float
  -- ^ Learning rate
  -> Int
  -- ^ No of iterations
  -> RNNWeights
  -- ^ Neural network weights
  -> FActivation
  -- ^ Activation function
  -> SerialT m ([Vector Float], Vector Float)
  -- ^ Data stream of inputs u1..uN and desired target yN
  -> m RNNWeights
sgdRNN hidden_layers lr n w0 fAct dataStream = iterN n step w0
  where
    step w = S.foldl' g w dataStream

    g :: RNNWeights
      -> ([Vector Float], Vector Float)
      -- ^ Training inputs and target
      -> RNNWeights
      -- ^ Trained neural network weights
    g w@(wI, wX, bX, wR) dta =
      let (dwI, dwX, dbX, dwR) = gradRNN hidden_layers w fAct dta
          wI1 = compute $ wI .- lr `_scale` dwI
          wX1 = compute $ wX .- lr `_scale` dwX
          bX1 = compute $ bX .- lr `_scale` dbX
          wR1 = compute $ wR .- lr `_scale` dwR
      in (wI1, wX1, bX1, wR1)

-- | Manually compute gradients for a two hidden layers approximation
gradRNN
  :: Int
  -- ^ Hidden layers for the recurrent layer approximation
  -> RNNWeights
  -- ^ Neural network weights
  -> FActivation
  -- ^ Activation function
  -> ([Vector Float], Vector Float)
  -- ^ Training example: inputs u1..uN and desired target yN
  -> RNNGradients
gradRNN hidden_layers w0 fAct dta = (dwI, dwX, dbX, dwR)
  where
    (u:u_:_, target) = dta :: ([Vector Float], Vector Float)

    f = getActivation fAct

    -- Initial weights
    (wI, wX, bX, wR) = w0

    -- Zero initial hidden state
    x0 = A.replicate Par (Sz (rows wX)) 0 :: Vector Float

    -- Step 1: from the first input to hidden layers
    u0 = u <# wI :: Vector Float

    -- Step 2: other inputs to hidden layers and between hidden layers:
    -- scan over `us` and prev value of hidden state `xprev`
    x1 = f (compute (u0 .+ x0 <# wX .+ bX)) :: Vector Float

    u1 = compute (u_ <# wI) :: Vector Float
    x2 = f (compute (u1 .+ x1 <# wX .+ bX)) :: Vector Float

    -- Step 3: readout
    y = x2 <# wR

    -- Loss gradient
    err = loss' (vec2m' y) (vec2m target)

    -- Step 3: readout gradient dwR
    dwR = linearW' (vec2m' x2) err
    dX3 = linearX' wR err

    -- Step 2: accumulate intermediate gradients as dwX and db
    dwX2 = linearW' (vec2m' x1) dX3
    dX2 = linearX' wX dX3
    dbX2 = bias' dX3

    dwX1 = linearW' (vec2m' x0) dX2
    dX1 = linearX' wX dX2
    dbX1 = bias' dX2

    dwX = compute (dwX2 .+ dwX1) :: Matrix Float
    dbX = compute (dbX2 .+ dbX1) :: Vector Float

    -- Step 1: dwI
    dwI = linearW' (vec2m' u) dX1

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

maxIndex :: (Ord a, Num b, Enum b) => [a] -> b
maxIndex xs = snd $ maximumBy (comparing fst) (zip xs [0..])

errors :: Eq lab => [(lab, lab)] -> [(lab, lab)]
errors = filter (uncurry (/=))
{-# SPECIALIZE errors :: [(Int, Int)] -> [(Int, Int)] #-}

accuracy :: (Eq a, Fractional acc) => [a] -> [a] -> acc
accuracy tgt pr = 100 * r
  where
    errNo = length $ errors (zip tgt pr)
    r = 1 - fromIntegral errNo / fromIntegral (length tgt)
{-# SPECIALIZE accuracy :: [Int] -> [Int] -> Float #-}

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
