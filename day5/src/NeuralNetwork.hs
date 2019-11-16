-- |= Convolutional Neural Network Building Blocks
--
-- Note that some functions have been updated w.r.t massiv-0.4.3.0,
-- most notably changed Data.Massiv.Array.Numeric
--
-- This work was largely inspired by
-- https://github.com/mstksg/backprop/blob/master/samples/backprop-mnist.lhs

{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE ViewPatterns #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

module NeuralNetwork
  ( LeNet
  , Vector
  , Matrix
  , Volume
  , Volume4
  , sigmoid
  , relu
  , relu_
  , conv2d
  , conv2d_
  , conv2d'
  , conv2d''
  , maxpool
  , maxpool_
  , softmax_
  , flatten
  , linear
  , forward
  , lenet
  , noPad2
  , (~>)

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
  , randLinear
  , randConv2d
  , randNetwork
  , rand
  , randn
  , iterN
  , br
  ) where

import           Control.Applicative ( liftA2 )
import           Control.DeepSeq ( NFData )
import           Control.Monad ( replicateM, foldM )
import           Data.List ( foldl', maximumBy )
import           Data.Massiv.Array hiding ( map, zip, zipWith, flatten )
import qualified Data.Massiv.Array as A
import           Data.Ord
import           GHC.Generics ( Generic )
import           Lens.Micro
import           Lens.Micro.TH
import           Numeric.Backprop
import           Numeric.Backprop.Class ( addNum )
import           Numeric.OneLiner
import           Streamly
import qualified Streamly.Prelude as S
import qualified System.Random.MWC as MWC
import           System.Random.MWC ( createSystemRandom )
import           System.Random.MWC.Distributions ( standard )

-- TODO hlint

-- Note that images are volumes of channels x width x height, whereas
-- mini-batches are volumes-4 of batch size x channels x width x height.
-- Similarly, convolutional filter weights are volumes-4 of
-- out channels x in channels x kernel width x kernel height.
type Vector a = Array U Ix1 a
type Matrix a = Array U Ix2 a
type Volume a = Array U Ix3 a
type Volume4 a = Array U Ix4 a

-- | Learnable neural network parameters.
-- Fully-connected layer weights.
data Linear a = Linear { _w :: !(Matrix a)
                       , _b :: !(Vector a)
                       }
  deriving (Show, Generic)

-- | Convolutional layer weights. We omit biases for simplicity.
data Conv2d a = Conv2d { _kernels :: !(Volume4 a) }
  deriving (Show, Generic)

instance NFData (Linear a)
makeLenses ''Linear

instance NFData (Conv2d a)
makeLenses ''Conv2d

data LeNet a =
    LeNet { _conv1 :: !(Conv2d a)
          , _conv2 :: !(Conv2d a)
          , _fc1 :: !(Linear a)
          , _fc2 :: !(Linear a)
          , _fc3 :: !(Linear a)
          }
  deriving (Show, Generic)

makeLenses ''LeNet

sameConv2d :: Reifies s W
    => BVar s (Conv2d Float)
    -> BVar s (Volume4 Float)
    -> BVar s (Volume4 Float)
sameConv2d = conv2d (Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0))

validConv2d :: Reifies s W
    => BVar s (Conv2d Float)
    -> BVar s (Volume4 Float)
    -> BVar s (Volume4 Float)
validConv2d = conv2d noPad2

lenet
    :: (Reifies s W)
    => BVar s (LeNet Float)
    -> Volume4 Float  -- ^ Batch of MNIST images
    -> BVar s (Matrix Float)
lenet l = constVar

          -- Feature extractor
          -- Layer (layer group) #1
          ~> sameConv2d (l ^^. conv1)
          ~> relu
          ~> maxpool
          -- Layer #2
          ~> sameConv2d (l ^^. conv2)
          -- ~> validConv2d (l ^^. conv2)
          ~> relu
          ~> maxpool

          ~> flatten

          -- Classifier
          -- Layer #3
          ~> linear (l ^^. fc1)
          ~> relu
          -- Layer #4
          ~> linear (l ^^. fc2)
          ~> relu
          -- Layer #5
          ~> linear (l ^^. fc3)
{-# INLINE lenet #-}

infixl 9 ~>
(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f
{-# INLINE (~>) #-}

noPad2 :: Padding Ix2 Float
noPad2 = Padding (Sz2 0 0) (Sz2 0 0) (Fill 0.0)

type ConvNet = LeNet

-- We would like to be able to perform arithmetic
-- operations over parameters, e.g. in SDG implementation.
-- Therefore, we define the Num instance.
instance (Num a, Unbox a, Index ix) => Num (Array U ix a) where
    x + y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .+. delay y)
    x - y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .-. delay y)
    x * y       = maybe (error $ "Dimension mismatch " ++ show (size x, size y)) compute (delay x .*. delay y)
    negate x    = computeMap negate x
    -- Maybe define later, when we will actually need those
    abs         = error "Please define abs"
    signum      = error "Please define signum"
    fromInteger = error "Please define me"

instance (Num a, Unbox a) => Num (Conv2d a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Num a, Unbox a) => Num (Linear a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance ( Unbox a
         , Num a
         ) => Num (ConvNet a) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

-- instance (Num a, Unbox a) => Fractional (Conv2d a) where
--     (/)          = error "Please define Conv2d (/)"
--     recip        = error "Please define Conv2d recip"
--     fromRational = error "Please define Conv2d fromRational (introduce Conv2d a i o)"
--
-- instance (Num a, Unbox a) => Fractional (Linear a) where
--     (/)          = error "Please define Linear (/)"
--     recip        = error "Please define Linear recip"
--     fromRational = error "Please define Linear fromRational (introduce Linear a i o)"
--
-- instance ( Num a
--          , Unbox a
--          ) => Fractional (ConvNet a) where
--     (/)          = gDivide
--     recip        = gRecip
--     fromRational = gFromRational

instance (Num a, Unbox a) => Backprop (Conv2d a)
instance (Num a, Unbox a) => Backprop (Linear a)
instance (Num a, Unbox a) => Backprop (ConvNet a)

-- | 2D convolution that operates on a batch.
--
-- Padding is Ix2 because it is performed only w.r.t. width and height.
--
-- The stride is assumed to be 1, but this can be extended
-- in a straightforward manner with `computeWithStride`.
-- Do not forget to use strides 1 in batch and channel dimensions (Dim4, Dim3).
conv2d_
       :: Padding Ix2 Float  -- ^ Image plane padding
       -> Volume4 Float  -- ^ Weights
       -> Volume4 Float  -- ^ Batch of input features
       -> Volume4 Float  -- ^ Output features
conv2d_ (Padding (Sz szp1) (Sz szp2) be) w x = compute res
  where
    (Sz (cout :> cin :> x1 :. x2)) = size w
    -- Extract weights, add fake Dim4, and make stencil
    sten = makeCorrelationStencilFromKernel. resize' (Sz4 1 cin x1 x2). (w !>)
    {-# INLINE sten #-}
    -- Add zeroes in batch and channel dimensions
    pad4 = Padding (Sz (0 :> 0 :> szp1)) (Sz (0 :> 0 :> szp2)) be
    -- Note: we apply stencils on zero channel of *all* images in the batch
    base = computeAs U $ applyStencil pad4 (sten 0) x
    -- Again, stencils are applied simultaneously on all images for a given
    -- channel
    res = foldl' (\prev ch -> let conv = computeAs U $ applyStencil pad4 (sten ch) x
                              in computeAs U $ append' 3 prev conv) base [1..cout - 1]

-- | Input gradients
--
-- \[ dX = \delta (*) W_{flip}, \]
--
-- where (*) is cross-correlation. We also have to perform inner transpose
-- over the kernel volume since we are propagating errors in the backward
-- direction.
conv2d'
  :: Padding Ix2 Float -> Volume4 Float -> Volume4 Float -> Volume4 Float
conv2d' p w dz = res
  where
    res = conv2d_ p (compute $ rot180 $ compute $ transposeInner w) dz

-- | Kernel gradients
--
-- \[ dW = X * \delta \]
conv2d''
  :: Padding Ix2 Float -> Volume4 Float -> Volume4 Float -> Volume4 Float
conv2d'' p x dz = conv2d_ p d x
  where
    d = computeAs U $ transposeInner dz

rot180 :: Index ix => Array U ix Float -> Array D ix Float
rot180 = reverse' 1. reverse' 2
{-# INLINE rot180 #-}

-- | Differentiable 2D convolutional layer
conv2d :: Reifies s W
       => Padding Ix2 Float
       -> BVar s (Conv2d Float)
       -> BVar s (Volume4 Float)
       -> BVar s (Volume4 Float)
conv2d p = liftOp2. op2 $ \(Conv2d w) x ->
  (conv2d_ p w x, \dz -> let dw = conv2d'' p x dz
                             p1 = p  -- TODO
                             dx = conv2d' p1 w dz
                         in (Conv2d dw, dx) )

instance (Index ix, Num e, Unbox e) => Backprop (Array U ix e) where
    zero x = A.replicate Par (size x) 0
    add = addNum  -- Making use of Num Array instance
    one x = A.replicate Par (size x) 1

-- | Linear layer
linear :: Reifies s W
       => BVar s (Linear Float)
       -> BVar s (Matrix Float)
       -> BVar s (Matrix Float)
linear = liftOp2. op2 $ \(Linear w b) x ->
  let prod = maybe (error "Dimension mismatch") id (x |*| w)
      lin = maybe (error "Dimension mismatch") compute (delay prod .+. (b `rowsLike` x))
  in (lin, \dZ -> let dW = linearW' x dZ
                      dB = bias' dZ
                      dX = linearX' w dZ
                  in (Linear dW dB, dX)
     )

-- TODO: differentiable (|*|), (.+.), and (-) analogs

relu_ :: (Index ix, Unbox e, Ord e, Num e) => Array U ix e -> Array U ix e
relu_ = computeMap (max 0)

relu :: (Reifies s W, Index ix)
        => BVar s (Array U ix Float)
        -> BVar s (Array U ix Float)
relu = liftOp1. op1 $ \x ->
  (relu_ x, \dY ->
    let f x0 dy0 = if x0 <= 0
                      then 0
                      else dy0
     in compute $ A.zipWith f x dY)

-- | Elementwise sigmoid with gradients
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

maxpoolStencil2x2 :: Stencil Ix4 Float Float
maxpoolStencil2x2 = makeStencil (Sz4 1 1 2 2) 0 $ \ get -> let max4 x1 x2 x3 x4 = max (max (max x1 x2) x3) x4 in max4 <$> get (0 :> 0 :> 0 :. 0) <*> get (0 :> 0 :> 1 :. 1) <*> get (0 :> 0 :> 0 :. 1) <*> get (0 :> 0 :> 1 :. 0)

maxpool_ :: Volume4 Float -> Volume4 Float
maxpool_ = computeWithStride (Stride (1 :> 1 :> 2 :. 2)). applyStencil noPadding maxpoolStencil2x2

-- > testA = fromLists' Seq [[[[1..4],[5..8],[9..12],[13..16]]]] :: Array U Ix4 Float
--
-- > testA
-- Array U Seq (Sz (1 :> 1 :> 4 :. 4))
--   [ [ [ [ 1.0, 2.0, 3.0, 4.0 ]
--       , [ 5.0, 6.0, 7.0, 8.0 ]
--       , [ 9.0, 10.0, 11.0, 12.0 ]
--       , [ 13.0, 14.0, 15.0, 16.0 ]
--       ]
--     ]
--   ]
-- > maxpool_ testA
-- Array U Seq (Sz (1 :> 1 :> 2 :. 2))
--   [ [ [ [ 6.0, 8.0 ]
--       , [ 14.0, 16.0 ]
--       ]
--     ]
--   ]
-- > testB = resize' (Sz4 2 1 2 4) testA
-- > testB
-- Array U Seq (Sz (2 :> 1 :> 2 :. 4))
--   [ [ [ [ 1.0, 2.0, 3.0, 4.0 ]
--       , [ 5.0, 6.0, 7.0, 8.0 ]
--       ]
--     ]
--   , [ [ [ 9.0, 10.0, 11.0, 12.0 ]
--       , [ 13.0, 14.0, 15.0, 16.0 ]
--       ]
--     ]
--   ]
-- > maxpool_ testB
-- Array U Seq (Sz (2 :> 1 :> 1 :. 2))
--   [ [ [ [ 6.0, 8.0 ]
--       ]
--     ]
--   , [ [ [ 14.0, 16.0 ]
--       ]
--     ]
--   ]

maxpool :: Reifies s W
        => BVar s (Volume4 Float)
        -> BVar s (Volume4 Float)
maxpool = liftOp1. op1 $ \x ->
  let out = maxpool_ x
      s = Stride (1 :> 1 :> 2 :. 2)
      outUp = computeAs U $ upsample' s out
      maxima = A.zipWith (\a b -> if a == b then 1 else 0) outUp x
  in (out, \dz -> let dzUp = computeAs U $ upsample' s dz
                  in maybe (error "Dimensions") compute (maxima .*. delay dzUp))

-- Test maxpool gradients
-- testC :: Volume4 Float
-- testC =
--   let a0 = A.fromLists' Par [[1,7],[3,4]] :: Matrix Float
--       b = resize' (Sz4 1 1 2 2) a0
--    in gradBP maxpool b
-- Array U Par (Sz (1 :> 1 :> 2 :. 2))
--   [ [ [ [ 0.0, 1.0 ]
--       , [ 0.0, 0.0 ]
--       ]
--     ]
--   ]
--
flatten :: Reifies s W
        => BVar s (Volume4 Float)
        -> BVar s (Matrix Float)
flatten = liftOp1. op1 $ \x ->
  let sz0@(Sz (bs :> ch :> h :. w)) = size x
      sz = Sz2 bs (ch * h * w)
   in (resize' sz x, \dz -> resize' sz0 dz)

-- | A neural network may work differently in training and evaluation modes
data Phase = Train | Eval deriving (Show, Eq)

-- | Uniformly-distributed random numbers Array
rand
  :: (Mutable r ix e, MWC.Variate e) =>
     (e, e) -> Sz ix -> IO (Array r ix e)
rand rng sz = do
    gens <- initWorkerStates Par (\_ -> createSystemRandom)
    randomArrayWS gens sz (MWC.uniformR rng)

-- | Random values from the Normal distribution
randn :: forall e ix. (Fractional e, Index ix, Unbox e) => Sz ix -> IO (Array U ix e)
randn sz = do
    gens <- initWorkerStates Par (\_ -> createSystemRandom)
    r <- randomArrayWS gens sz standard :: IO (Array P ix Double)
    return (compute $ A.map realToFrac r)

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
{-# INLINE computeMap #-}

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

-- | Forward pass in a neural network (inference)
forward :: ConvNet Float -> Volume4 Float -> Matrix Float
forward net dta = evalBP (flip lenet dta) net

softmax_ :: Matrix Float -> Matrix Float
softmax_ x =
  let x0 = expA (delay x)
      x1 = computeAs U $ _sumCols x0  -- Note _sumCols, not _sumRows
      x2 = x1 `colsLike` x
  in maybe (error  "Inconsistent dimensions in softmax_") compute (x0 ./. x2)

crossEntropyLoss
  :: forall s. (Reifies s W)
  => Volume4 Float
  -> Matrix Float
  -> BVar s (ConvNet Float)
  -> BVar s (Matrix Float)
crossEntropyLoss x targ n = _ce y
  where
    y = lenet n x :: BVar s (Matrix Float)
    _ce :: BVar s (Matrix Float) -> BVar s (Matrix Float)
    -- -- Gradients only
    -- _ce = liftOp1. op1 $ \pred_ ->
    --   (undefined, \_ -> pred_ - targ)
    _ce pred_ = pred_ - targ_
    targ_ = constVar targ
{-# INLINE crossEntropyLoss #-}

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
    -- Iterate over all batches
    epochStep net = S.foldl' _trainStep net dataStream
    -- Update gradients based on a single batch
    _trainStep net (x, targ) = trainStep lr x targ net
    {-# INLINE _trainStep #-}

-- subtractGrad
--   :: (Num e, MonadThrow m, Source r1 ix e, Source r2 ix e) =>
--      e -> Array r1 ix e -> Array r2 ix e -> m (Array D ix e)
-- subtractGrad lr w dW = delay w .-. (lr *. delay dW)
--
-- subtractGradMaybe
--   :: (Mutable r ix e, Num e, Source r1 ix e, Source r2 ix e) =>
--      e -> Array r1 ix e -> Array r2 ix e -> Array r ix e
-- subtractGradMaybe lr w dW = maybe (error "Inconsistent dimensions") compute (subtractGrad lr w dW)

-- | Gradient descent step
trainStep
  :: Float  -- ^ Learning rate
  -> Volume4 Float  -- ^ Images batch
  -> Matrix Float  -- ^ Targets
  -> ConvNet Float  -- ^ Initial network
  -> ConvNet Float
trainStep lr !x !targ !n = n - (computeMap' (lr *) (gradBP (crossEntropyLoss x targ) n))
-- The problem is that realToFrac does not know about the shape.
-- This can be solved having that information on the type level.
-- Conv2d a i o k, i = in channels, o = out channels, k = square kernel size
-- and Linear a i o, i = inputs, o = outputs.
-- trainStep lr !x !targ !n = n - realToFrac lr * gradBP (loss x targ) n
{-# INLINE trainStep #-}

-- This could be definitely improved: see comments above (`trainStep`)
computeMap' :: (Float -> Float) -> LeNet Float -> LeNet Float
computeMap' f (LeNet { _conv1 = Conv2d k1
                     , _conv2 = Conv2d k2
                     , _fc1 = Linear w1 b1
                     , _fc2 = Linear w2 b2
                     , _fc3 = Linear w3 b3
                     }) = LeNet { _conv1 = Conv2d (computeMap f k1)
                                , _conv2 = Conv2d (computeMap f k2)
                                , _fc1 = Linear (computeMap f w1) (computeMap f b1)
                                , _fc2 = Linear (computeMap f w2) (computeMap f b2)
                                , _fc3 = Linear (computeMap f w3) (computeMap f b3)
                                }

-- | Strict left fold
iterN :: Monad m => Int -> (a -> m a) -> a -> m a
iterN n f x0 = foldM (\x _ -> f x) x0 [1..n]

-- | Generate random weights and biases
randLinear :: Sz2 -> IO (Linear Float)
randLinear sz@(Sz2 _ nout) = do
  _w <- setComp Par <$> _genWeights sz
  _b <- setComp Par <$> _genBiases nout
  return (Linear _w _b)
    where
      _genBiases n = randn (Sz n)

_genWeights :: Index ix => Sz ix -> IO (Array U ix Float)
_genWeights sz = do
    a <- randn sz
    return (compute $ k *. (delay a))
  where
    -- Weight scaling factor. Can also be dependent on `sz`.
    k = 0.01

-- | Generate random convolutional layer
randConv2d :: Sz4 -> IO (Conv2d Float)
randConv2d sz = do
  k <- setComp Par <$> _genWeights sz
  return (Conv2d k)

randNetwork :: IO (ConvNet Float)
randNetwork = do
  _conv1 <- randConv2d (Sz4 3 1 5 5)
  _conv2 <- randConv2d (Sz4 3 3 5 5)
  let [i, h1, h2, o] = [3 * 7 * 7, 120, 84, 10]
  _fc1 <- randLinear (Sz2 i h1)
  _fc2 <- randLinear (Sz2 h1 h2)
  _fc3 <- randLinear (Sz2 h2 o)
  return $
    LeNet { _conv1 = _conv1
          , _conv2 = _conv2
          , _fc1 = _fc1
          , _fc2 = _fc2
          , _fc3 = _fc3
          }

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
