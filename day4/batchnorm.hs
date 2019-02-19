{-# LANGUAGE BangPatterns                         #-}
{-# LANGUAGE DataKinds                            #-}
{-# LANGUAGE DeriveGeneric                        #-}
{-# LANGUAGE FlexibleContexts                     #-}
{-# LANGUAGE FlexibleInstances                    #-}
{-# LANGUAGE MultiParamTypeClasses                #-}
{-# LANGUAGE PartialTypeSignatures                #-}
{-# LANGUAGE ScopedTypeVariables                  #-}
{-# LANGUAGE TemplateHaskell                      #-}
{-# LANGUAGE TypeApplications                     #-}
{-# LANGUAGE TypeOperators                        #-}
{-# LANGUAGE TypeSynonymInstances                 #-}
{-# LANGUAGE ViewPatterns                         #-}
{-# LANGUAGE UndecidableInstances                 #-}
{-# OPTIONS_GHC -fno-warn-orphans                 #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

import           Numeric.Backprop as BP
import           Control.DeepSeq
import           Control.Exception
import           Control.Monad
import           Control.Monad.IO.Class
import           Control.Monad.Trans.Class
import           Control.Monad.Trans.Maybe
import           Control.Monad.Trans.State
import           Data.Bitraversable
import           Data.Conduit
import           Data.Default
import           Data.IDX
import           Data.List ( foldl' )
import           Data.List.Split
import           Data.Primitive.MutVar
import           Data.Time
import           Data.Traversable
import           Data.Tuple
import           GHC.TypeNats
import           GHC.Generics                        (Generic)
import           Lens.Micro
import           Lens.Micro.TH
import           Numeric.LinearAlgebra.Static.Backprop as BP
import           Numeric.OneLiner
import           Numeric.Opto hiding ( (<.>) )
import           System.Environment
import           System.FilePath
import           Text.Printf
import qualified Data.Conduit.Combinators              as C
import qualified Data.Vector                           as V
import qualified Data.Vector.Generic                   as VG
import qualified Data.Vector.Unboxed                 as VU
import qualified Numeric.LinearAlgebra                 as HM
import qualified Numeric.LinearAlgebra.Static          as H
import qualified System.Random.MWC                     as MWC
import qualified System.Random.MWC.Distributions       as MWC

import           Model ( batchMean, batchVar )

a0 = H.fromList [1,2] :: H.R 2
v0 = H.fromList [3,10] :: H.R 2

runBatchNorm
  :: (Reifies s W, KnownNat i)
  => BVar s [R i] -> BVar s [R i]
runBatchNorm batch = collectVar $ f (sequenceVar batch)
  where
    f = map (\v -> (v - mu) / sqrt (var + epsilon))
    mu = batchMean batch
    var = batchVar batch
    epsilon = 1e-12

batch :: [H.R 2]
batch = [a0, v0]

testBatchnorm = do
  let res = evalBP runBatchNorm batch
  mapM_ (print. H.extract) res

data Layer i o =
    Layer { _lWeights :: !(L o i)
          , _lBiases  :: !(R o)
          }
  deriving (Show, Generic)

instance NFData (Layer i o)
makeLenses ''Layer

-- And a type for a simple feed-forward network with two hidden layers:

data Network i h1 h2 o =
    Net { _nLayer1 :: !(Layer i  h1)
        , _nLayer2 :: !(Layer h1 h2)
        , _nLayer3 :: !(Layer h2 o)
        }
  deriving (Show, Generic)

instance NFData (Network i h1 h2 o)
makeLenses ''Network

instance (KnownNat i, KnownNat o) => Num (Layer i o) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance ( KnownNat i
         , KnownNat h1
         , KnownNat h2
         , KnownNat o
         ) => Num (Network i h1 h2 o) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (KnownNat i, KnownNat o) => Fractional (Layer i o) where
    (/)          = gDivide
    recip        = gRecip
    fromRational = gFromRational

instance ( KnownNat i
         , KnownNat h1
         , KnownNat h2
         , KnownNat o
         ) => Fractional (Network i h1 h2 o) where
    (/)          = gDivide
    recip        = gRecip
    fromRational = gFromRational

instance (KnownNat i, KnownNat o) => Backprop (Layer i o)
instance (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o) => Backprop (Network i h1 h2 o)

runLayer
    :: (KnownNat i, KnownNat o, Reifies s W)
    => BVar s (Layer i o)
    -> BVar s (R i)
    -> BVar s (R o)
runLayer l x = (l ^^. lWeights) #> x + (l ^^. lBiases)
{-# INLINE runLayer #-}

softMax :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
softMax x = konst (1 / sumElements expx) * expx
  where
    expx = exp x
{-# INLINE softMax #-}

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

runNetwork
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => BVar s (Network i h1 h2 o)
    -> R i
    -> BVar s (R o)
runNetwork n = softMax
             . runLayer (n ^^. nLayer3)
             . sigmoid
             . runLayer (n ^^. nLayer2)
             . sigmoid
             . runLayer (n ^^. nLayer1)
             . constVar
{-# INLINE runNetwork #-}

crossEntropy
    :: (KnownNat n, Reifies s W)
    => R n
    -> BVar s (R n)
    -> BVar s Double
crossEntropy targ res = -(log res BP.<.> constVar targ)
{-# INLINE crossEntropy #-}

netErr
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => R i
    -> R o
    -> BVar s (Network i h1 h2 o)
    -> BVar s Double
netErr x targ n = crossEntropy targ (runNetwork n x)
{-# INLINE netErr #-}

trainStep
    :: forall i h1 h2 o. (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => Double             -- ^ learning rate
    -> R i                -- ^ input
    -> R o                -- ^ target
    -> Network i h1 h2 o  -- ^ initial network
    -> Network i h1 h2 o
trainStep r !x !targ !n = n - realToFrac r * gradBP (netErr x targ) n
{-# INLINE trainStep #-}

trainList
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => Double             -- ^ learning rate
    -> [(R i, R o)]       -- ^ input and target pairs
    -> Network i h1 h2 o  -- ^ initial network
    -> Network i h1 h2 o
trainList r = flip $ foldl' (\n (x,y) -> trainStep r x y n)
{-# INLINE trainList #-}

testNet
    :: forall i h1 h2 o. (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => [(R i, R o)]
    -> Network i h1 h2 o
    -> Double
testNet xs n = sum (map (uncurry test) xs) / fromIntegral (length xs)
  where
    test :: R i -> R o -> Double          -- test if the max index is correct
    test x (H.extract->t)
        | HM.maxIndex t == HM.maxIndex (H.extract r) = 1
        | otherwise                                = 0
      where
        r :: R o
        r = evalBP (`runNetwork` x) n

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
    Just train <- loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
    Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
    putStrLn "Loaded data."
    net0 <- MWC.uniformR @(Network 784 300 100 10) (-0.5, 0.5) g
    flip evalStateT net0 . forM_ [1..] $ \e -> do
      train' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList train) g
      liftIO $ printf "[Epoch %d]\n" (e :: Int)

      forM_ ([1..] `zip` chunksOf batch train') $ \(b, chnk) -> StateT $ \n0 -> do
        printf "(Batch %d)\n" (b :: Int)

        t0 <- getCurrentTime
        n' <- evaluate . force $ trainList rate chnk n0
        t1 <- getCurrentTime
        printf "Trained on %d points in %s.\n" batch (show (t1 `diffUTCTime` t0))

        let trainScore = testNet chnk n'
            testScore  = testNet test n'
        printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
        printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)

        return ((), n')
  where
    rate  = 0.02
    batch = 5000

loadMNIST
    :: FilePath
    -> FilePath
    -> IO (Maybe [(R 784, R 10)])
loadMNIST fpI fpL = runMaybeT $ do
    i <- MaybeT          $ decodeIDXFile       fpI
    l <- MaybeT          $ decodeIDXLabelsFile fpL
    d <- MaybeT . return $ labeledIntData l i
    r <- MaybeT . return $ for d (bitraverse mkImage mkLabel . swap)
    liftIO . evaluate $ force r
  where
    mkImage :: VU.Vector Int -> Maybe (R 784)
    mkImage = H.create . VG.convert . VG.map (\i -> fromIntegral i / 255)
    mkLabel :: Int -> Maybe (R 10)
    mkLabel n = H.create $ HM.build 10 (\i -> if round i == n then 1 else 0)

instance KnownNat n => MWC.Variate (R n) where
    uniform g = H.randomVector <$> MWC.uniform g <*> pure H.Uniform
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance (KnownNat m, KnownNat n) => MWC.Variate (L m n) where
    uniform g = H.uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance (KnownNat i, KnownNat o) => MWC.Variate (Layer i o) where
    uniform g = Layer <$> MWC.uniform g <*> MWC.uniform g
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance ( KnownNat i
         , KnownNat h1
         , KnownNat h2
         , KnownNat o
         )
      => MWC.Variate (Network i h1 h2 o) where
    uniform g = Net <$> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g
    uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g
