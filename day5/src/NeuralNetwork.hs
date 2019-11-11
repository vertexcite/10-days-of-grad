-- |= Neural Network Building Blocks
--
-- Note that some functions have been updated w.r.t massiv-0.4.3.0,
-- most notably changed Data.Massiv.Array.Numeric

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}

module NeuralNetwork
  ( ConvNet
  , Layer (..)
  , Vector
  , Matrix
  , Volume
  , Volume4
  , sigmoid
  , relu
  , conv2d
  , maxpool
  , NeuralNetwork.flatten
  , linear
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
import           Numeric.Backprop

-- Note that images are volumes of channels x width x height, whereas
-- mini-batches are volumes-4 of batch size x channels x width x height.
-- Similarly, convolutional filter weights are volumes-4 of
-- out channels x in channels x kernel width x kernel height.
type Vector a = Array U Ix1 a
type Matrix a = Array U Ix2 a
type Volume a = Array U Ix3 a
type Volume4 a = Array U Ix4 a

-- Cross-correlation
conv2d_ :: Padding Ix3 Float  -- ^ Padding
        -> Stride Ix3
        -> Array U Ix4 Float  -- ^ Weights
        -> Array U Ix3 Float  -- ^ Input features
        -> Array U Ix3 Float  -- ^ Output features
conv2d_ padding stride w x = compute $ A.concat' (Dim 3) results
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    stencils = map (makeCorrelationStencilFromKernel. (w !>)) [0..cout - 1]
    results :: [Array U Ix3 Float]
    results = map (\s -> computeWithStride stride $ applyStencil padding s x) stencils

-- Actual convolution
conv2d__ :: Padding Ix3 Float  -- ^ Padding
        -> Stride Ix3
        -> Array U Ix4 Float  -- ^ Weights
        -> Array U Ix3 Float  -- ^ Input features
        -> Array U Ix3 Float  -- ^ Output features
conv2d__ padding stride w x = compute $ A.concat' (Dim 3) results
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    stencils = map (makeConvolutionStencilFromKernel. (w !>)) [0..cout - 1]
    results :: [Array U Ix3 Float]
    results = map (\s -> computeWithStride stride $ applyStencil padding s x) stencils

-- TODO: work on Volume4 mini-batches
-- | 2D convolution with derivatives
conv2d :: Reifies s W
       => Padding Ix3 Float
       -> Stride Ix3
       -> BVar s (Volume4 Float)
       -> BVar s (Volume Float)
       -> BVar s (Volume Float)
conv2d p s = liftOp2. op2 $ \w x ->
  (conv2d_ p s w x, \dz -> let dw = undefined
                               dx = undefined
                            in (dw, dx) )

instance (Index ix, Num e, Unbox e) => Backprop (Array U ix e) where
    zero x = A.replicate Par (size x) 0
    add x y = maybe (error "Dimension mismatch") compute (delay x .+. delay y)
    one x = A.replicate Par (size x) 1

-- TODO: refactor to a composition of
-- new differentiable |*| and .+. operators
linear :: Reifies s W
       => BVar s (Matrix Float)
       -> BVar s (Vector Float)
       -> BVar s (Matrix Float)
       -> BVar s (Matrix Float)
linear = liftOp3. op3 $ \w b x ->
  let prod = maybe (error "Dimension mismatch") id (x |*| w)
      lin = maybe (error "Dimension mismatch") compute (delay prod .+. (b `rowsLike` x))
  in (lin, \dZ -> let dW = linearW' x dZ
                      dB = bias' dZ
                      dX = linearX' w dZ
                  in (dW, dB, dX)
     )

relu :: (Reifies s W, Index ix)
        => BVar s (Array U ix Float)
        -> BVar s (Array U ix Float)
relu = liftOp1. op1 $ \x ->
  (computeMap (max 0.0) x, \dY ->
    let f x0 dy0 = if x0 <= 0
                      then 0
                      else dy0
     in compute $ A.zipWith f x dY)

-- | Elementwise sigmoid and its derivative
sigmoid :: forall s ix. (Reifies s W, Index ix)
        => BVar s (Array U ix Float)
        -> BVar s (Array U ix Float)
sigmoid = liftOp1. op1 $ \x ->
    let y = computeMap f x
    in (y, \dY ->
        let ones = delay $ one x
            y' = delay y
        in either throw compute $ do
            e1 <- ones .-. y'
            e2 <- y' .*. e1
            delay dY .*. e2
       )
  where
    f x = recip $ 1.0 + exp (-x)

-- TODO: operate on Volume4 mini-batches
maxpool = undefined

flatten :: Reifies s W
        => BVar s (Volume4 Float)
        -> BVar s (Matrix Float)
flatten = liftOp1. op1 $ \x ->
  let sz0@(Sz (bs :> ch :> h :. w)) = size x
      sz = Sz2 bs (ch * h * w)
   in (resize' sz x, \dz -> resize' sz0 dz)

data Layer a = Layer a
type ConvNet a = [Layer a]
data Grad a = Grad a

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

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
  -> (Matrix Float, [Grad Float])
  -- ^ NN computation from forward pass and weight gradients
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
sgd lr n net0 dataStream = undefined
-- sgd lr n net0 dataStream = iterN n epochStep net0
--   where
--     epochStep net = S.foldl' g net dataStream
--
--     g :: ConvNet Float
--       -> (Volume4 Float, Matrix Float)
--       -> ConvNet Float
--     g net dta =
--       let (_, dW) = pass Train net dta
--       in (zipWith f net dW)
--
--     f :: Layer Float -> Grad Float -> Layer Float
--
--     -- Update Linear layer weights
--     f (Linear w b) (LinearGrad dW dB) =
--       let w1 = subtractGradMaybe lr w dW
--           b1 = subtractGradMaybe lr b dB
--       in Linear w1 b1
--
--     f (Conv2d w) (Conv2dGrad dW) = Conv2d (subtractGradMaybe lr w dW)
--
--     -- No parameters to change
--     f layer NoGrad = layer
--
--     f _ _ = error "Layer/gradients mismatch"

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
