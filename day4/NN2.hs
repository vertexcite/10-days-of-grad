#!/usr/bin/env stack
-- stack --install-ghc ghci --resolver lts-11.9 --package backprop-0.2.2.0 \
-- --package random --package hmatrix-backprop-0.1.2.1 --package statistics \
-- --package lens --package one-liner-instances --package split --package \
-- ghc-typelits-natnormalise --package ghc-typelits-knownnat --package hmatrix \
-- --package hmatrix-vector-sized --package microlens --package vector-sized \
-- --package transformers --package type-combinators -- -O2 NN2.hs
--
-- | Fully-connected neural network with automatic differentiation
--
-- Based on https://blog.jle.im/entry/purely-functional-typed-models-1.html

{-# LANGUAGE DataKinds                                #-}
{-# LANGUAGE DeriveGeneric                            #-}
{-# LANGUAGE FlexibleContexts                         #-}
{-# LANGUAGE FlexibleInstances                        #-}
{-# LANGUAGE GADTs                                    #-}
{-# LANGUAGE LambdaCase                               #-}
{-# LANGUAGE MultiParamTypeClasses                    #-}
{-# LANGUAGE PartialTypeSignatures                    #-}
{-# LANGUAGE PatternSynonyms                          #-}
{-# LANGUAGE RankNTypes                               #-}
{-# LANGUAGE ScopedTypeVariables                      #-}
{-# LANGUAGE TypeApplications                         #-}
{-# LANGUAGE TypeInType                               #-}
{-# LANGUAGE TypeOperators                            #-}
{-# LANGUAGE ViewPatterns                             #-}
{-# OPTIONS_GHC -fno-warn-orphans                     #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures     #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise       #-}
{-# OPTIONS_GHC -fwarn-redundant-constraints          #-}

import           Control.Monad.Trans.State
import           Data.Bifunctor
import           Data.Foldable
import           Data.List
import           Data.List.Split
import           Data.Tuple
import           Data.Type.Option
import           GHC.Generics                          (Generic)
import           GHC.TypeNats
import           Lens.Micro hiding                     ((&))
import           Numeric.Backprop
import           Numeric.LinearAlgebra.Static.Backprop
import           Numeric.LinearAlgebra.Static.Vector
import           Numeric.OneLiner
import           System.Random
import qualified Data.Vector.Storable.Sized            as SVS
import qualified Numeric.LinearAlgebra                 as HU
import qualified Numeric.LinearAlgebra.Static          as H
import qualified Prelude.Backprop                      as B
import qualified Numeric.LinearAlgebra as LA


{-
data a :& b = !a :& !b
  deriving (Show, Generic)
infixr 2 :&

type Model p a b = forall z. Reifies z W
                => BVar z p
                -> BVar z a
                -> BVar z b

squaredErrorGrad
    :: (Backprop p, Backprop b, Num b)
    => Model p a b      -- ^ Model
    -> a                -- ^ Observed input
    -> b                -- ^ Observed output
    -> p                -- ^ Parameter guess
    -> p                -- ^ Gradient
squaredErrorGrad f x targ = gradBP $ \p ->
    (f p (auto x) - auto targ) ^ 2


crossEntropyGrad
    :: (Backprop p, Backprop b, Floating b)
    => Model p a b      -- ^ Model
    -> a                -- ^ Observed input
    -> b                -- ^ Observed output
    -> p                -- ^ Parameter guess
    -> p                -- ^ Gradient
crossEntropyGrad f x targ = gradBP $ \p ->
    let y = f p (auto x)
        y1 = auto targ
    in (-y1 * log y - (1 - y1) * log (1 - y))

trainModel
    :: (Fractional p, Backprop p, Backprop b, Floating b)
    => Model p a b      -- ^ model to train
    -> p                -- ^ initial parameter guess
    -> [(a,b)]          -- ^ list of observations
    -> p                -- ^ updated parameter guess
trainModel f = foldl' $ \p (x,y) -> p - 0.001 * crossEntropyGrad f x y p

-- trainModelIO
--     :: (Fractional p, Backprop p, Backprop b, Random p, Floating b)
--     => Model p a b      -- ^ model to train
--     -> [(a,b)]          -- ^ list of observations
--     -> IO p             -- ^ parameter guess
-- trainModelIO m xs = do
--     -- Problem with weight generation:
--     -- 1) Should be from Gaussian distribution
--     -- 2) Should be scaled w.r.t. number of inputs (Xavier init)
--     p0 <- (/ 10) . subtract 0.5 <$> randomIO
--     return $ trainModel m p0 xs

logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))

feedForward
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForward (w :&& b) x = w #> x + b

feedForwardLog'
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardLog' wb = logistic . feedForward wb

feedForwardTanh
    :: (KnownNat i, KnownNat o)
    => Model (L o i :& R o) (R i) (R o)
feedForwardTanh wb = tanh . feedForward wb

makeCircles ::
  Int -> Double -> Double -> IO [(R 2, R 1)]
makeCircles m factor noise = do
  let rand' n = (LA.scale (2 * pi)) <$> LA.rand n 1
      m1 = m `div` 2
      m2 = m - (m `div` 2)

  r1 <- rand' m1
  r2 <- rand' m2

  let outerX = cos r1
      outerY = sin r1
      innerX = LA.scale factor $ cos r2
      innerY = LA.scale factor $ sin r2
      -- Merge them all

      -- TODO: generate random lists without LA
      flatten1 = LA.toList. LA.flatten
      outerX' = flatten1 outerX
      outerY' = flatten1 outerY
      innerX' = flatten1 innerX
      innerY' = flatten1 innerY

      x1 = zipWith H.vec2 outerX' outerY'
      x2 = zipWith H.vec2 innerX' innerY'

      y1 = map (\_ -> 0) x1
      y2 = map (\_ ->  1) x2

  return $ zip (x1 ++ x2) (y1 ++ y2)

(<~)
    :: (Backprop p, Backprop q)
    => Model  p       b c
    -> Model       q  a b
    -> Model (p :& q) a c
(f <~ g) (p :&& q) = f p . g q
infixr 8 <~

testTrain3 :: _ -> IO [R 1]
testTrain3 samps = do
    w1 <- H.randn :: IO (L 128 2)
    mb1 <- H.randn :: IO (L 128 1)
    w2 <- H.randn :: IO (L 1 128)
    mb2 <- H.randn :: IO (L 1 1)
    let b1 = H.uncol mb1 :: R 128
    let b2 = H.uncol mb2 :: R 1

    let p0 = ((w2 :& b2) :& (w1 :& b1))

    let trained = trainModel model p0 (take 200000 (cycle samps))
    -- trained <- trainModelIO model (take 200000 (cycle samps))

    return [ evalBP2 model trained r | (r, _) <- samps ]

model :: Model _ (R 2) (R 1)
model = feedForwardLog' @128 @1 <~ feedForwardTanh @2 @128


main :: IO ()
main = do
  samps' <- makeCircles 200 0.6 0.1
  -- mapM_ (\((a,b),c) -> putStrLn $ show a ++ " " ++ show b) samps'
  mapM_ (putStrLn. show) =<< testTrain3 samps'


pattern (:&&) :: (Backprop a, Backprop b, Reifies z W)
              => BVar z a -> BVar z b -> BVar z (a :& b)
pattern x :&& y <- (\xy -> (xy ^^. t1, xy ^^. t2)->(x, y))
  where
    (:&&) = isoVar2 (:&) (\case x :& y -> (x, y))
{-# COMPLETE (:&&) #-}

t1 :: Lens (a :& b) (a' :& b) a a'
t1 f (x :& y) = (:& y) <$> f x

t2 :: Lens (a :& b) (a :& b') b b'
t2 f (x :& y) = (x :&) <$> f y

instance (Num a, Num b) => Num (a :& b) where
    (+)         = gPlus
    (-)         = gMinus
    (*)         = gTimes
    negate      = gNegate
    abs         = gAbs
    signum      = gSignum
    fromInteger = gFromInteger

instance (Fractional a, Fractional b) => Fractional (a :& b) where
    (/) = gDivide
    recip = gRecip
    fromRational = gFromRational

instance (Random a, Random b) => Random (a :& b) where
    random g0 = (x :& y, g2)
      where
        (x, g1) = random g0
        (y, g2) = random g1
    randomR (x0 :& y0, x1 :& y1) g0 = (x :& y, g2)
      where
        (x, g1) = randomR (x0, x1) g0
        (y, g2) = randomR (y0, y1) g1

instance (Backprop a, Backprop b) => Backprop (a :& b)

uncurryT
    :: (Backprop a, Backprop b, Reifies z W)
    => (BVar z a -> BVar z b -> BVar z c)
    -> BVar z (a :& b)
    -> BVar z c
uncurryT f x = f (x ^^. t1) (x ^^. t2)

instance (KnownNat n, KnownNat m) => Random (L n m) where
    random = runState . fmap vecL $ SVS.replicateM (state random)
    randomR (xs,ys) = runState . fmap vecL $ SVS.zipWithM (curry (state . randomR))
        (lVec xs) (lVec ys)

instance (KnownNat n) => Random (R n) where
    random = runState $ vecR <$> SVS.replicateM (state random)
    randomR (xs,ys) = runState . fmap vecR $ SVS.zipWithM (curry (state . randomR))
        (rVec xs) (rVec ys)

-}

main = return ()
