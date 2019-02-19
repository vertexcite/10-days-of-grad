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

linearLayer
    :: (KnownNat i, KnownNat o, Reifies s W)
    => BVar s (Layer i o)  -- ^ Weights
    -> BVar s [R i]        -- ^ Input batch
    -> BVar s [R o]        -- ^ Output batch
linearLayer l = withLayer f
  where
    f = (\x -> (l ^^. lWeights) #> x + (l ^^. lBiases))
{-# INLINE linearLayer #-}

batchNormLayer
  :: (Reifies s W, KnownNat i)
  => BVar s [R i] -> BVar s [R i]
batchNormLayer batch = withLayer f batch
  where
    f v = (v - mu) / sqrt (var + epsilon)
    mu = batchMean batch
    var = batchVar batch
    epsilon = 1e-12
{-# INLINE batchNormLayer #-}

-- TODO batchNormAffineLayer

testBatchnorm = do
  let a0 = H.fromList [1,10] :: H.R 2
      v0 = H.fromList [3,2] :: H.R 2
      batch0 :: [H.R 2]
      batch0 = [a0, v0]

  let res = evalBP batchNormLayer batch0
  mapM_ (print. H.extract) res
  -- [-1, 1]
  -- [1, -1]

sigmoid :: Floating a => a -> a
sigmoid x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

softMax :: (KnownNat n, Reifies s W) => BVar s (R n) -> BVar s (R n)
softMax x = konst (1 / sumElements expx) * expx
  where
    expx = exp x
{-# INLINE softMax #-}

withLayer
  :: (Reifies s W, KnownNat i, KnownNat o)
  => (BVar s (R i) -> BVar s (R o))
  -> BVar s [R i] -> BVar s [R o]
withLayer f = collectVar. map f. sequenceVar

inputs :: (KnownNat i, Reifies s W)
       => [R i]
       -> BVar s [R i]
inputs = collectVar. map constVar

-- Run network over batches of data
runNetwork1
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => BVar s (Network i h1 h2 o)
    -> [R i]
    -> BVar s [R o]
runNetwork1 n =
             -- Layer #3
             withLayer softMax
             . batchNormLayer
             . linearLayer (n ^^. nLayer3)

             -- Layer #2
             . withLayer sigmoid
             . batchNormLayer
             . linearLayer (n ^^. nLayer2)

             -- Layer #1
             . withLayer sigmoid
             . batchNormLayer
             . linearLayer (n ^^. nLayer1)

             . inputs
{-# INLINE runNetwork1 #-}

-- | No batch normalization
runNetwork0
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => BVar s (Network i h1 h2 o)
    -> [R i]
    -> BVar s [R o]
runNetwork0 n =
             -- Layer #3
             withLayer softMax
             . linearLayer (n ^^. nLayer3)

             -- Layer #2
             . withLayer sigmoid
             . linearLayer (n ^^. nLayer2)

             -- Layer #1
             . withLayer sigmoid
             . linearLayer (n ^^. nLayer1)

             . inputs
{-# INLINE runNetwork0 #-}

runNetwork
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => BVar s (Network i h1 h2 o)
    -> [R i]
    -> BVar s [R o]
runNetwork = runNetwork1

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

crossEntropy
    :: (KnownNat n, Reifies s W)
    => R n
    -> BVar s (R n)
    -> BVar s Double
crossEntropy targ res = -(log res BP.<.> constVar targ)
{-# INLINE crossEntropy #-}

netErr
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o, Reifies s W)
    => [R i]
    -> [R o]
    -> BVar s (Network i h1 h2 o)
    -> BVar s Double
netErr xs targ n = sum $ zipWith crossEntropy targ $ sequenceVar (runNetwork n xs)
{-# INLINE netErr #-}

trainStep
    :: forall i h1 h2 o. (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => Double             -- ^ learning rate
    -> [R i]                -- ^ input batch
    -> [R o]                -- ^ targets
    -> Network i h1 h2 o  -- ^ initial network
    -> Network i h1 h2 o
trainStep r !x !targ !n = n - realToFrac r * gradBP (netErr x targ) n
{-# INLINE trainStep #-}

trainBatch
    :: (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => Double             -- ^ learning rate
    -> [(R i, R o)]       -- ^ input and target pairs
    -> Network i h1 h2 o  -- ^ initial network
    -> Network i h1 h2 o
trainBatch r dta n =
  let (xs, tgt) = unzip dta
  in trainStep r xs tgt n
{-# INLINE trainBatch #-}

-- | Test network on a batch of samples
testNet
    :: forall i h1 h2 o. (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
    => [(R i, R o)]
    -> Network i h1 h2 o
    -> Double
testNet xs n = sum (zipWith test rs tgt) / fromIntegral (length rs)
  where
    dta = map fst xs
    tgt = map snd xs

    rs :: [R o]
    rs = evalBP (`runNetwork` dta) n

    test :: R o -> R o -> Double          -- test if the max index is correct
    test r (H.extract->t)
        | HM.maxIndex t == HM.maxIndex (H.extract r) = 1
        | otherwise = 0

main :: IO ()
main = MWC.withSystemRandom $ \g -> do
    Just train <- loadMNIST "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
    Just test  <- loadMNIST "data/t10k-images-idx3-ubyte"  "data/t10k-labels-idx1-ubyte"
    putStrLn "Loaded data."
    net0 <- MWC.uniformR @(Network 784 300 100 10) (-0.5, 0.5) g
    flip evalStateT net0 . forM_ [1..] $ \e -> do
      train' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList train) g
      liftIO $ printf "[Epoch %d]\n" (e :: Int)

      forM_ ([1..] `zip` chunksOf batchSize train') $ \(b, chnk) -> StateT $ \n0 -> do
        printf "(Batch %d)\n" (b :: Int)

        t0 <- getCurrentTime
        n' <- evaluate . force $ trainBatch rate chnk n0
        t1 <- getCurrentTime
        printf "Trained on %d points in %s.\n" batchSize (show (t1 `diffUTCTime` t0))

        let trainScore = testNet chnk n'
            testScore  = testNet test n'
        printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
        printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)

        return ((), n')
  where
    rate  = 0.02
    batchSize = 512

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
