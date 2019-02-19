-- Simplified version of
-- https://github.com/mstksg/backprop-learn/blob/master/src/Backprop/Learn/Model/Function.hs

{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE DeriveDataTypeable    #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE GADTs                 #-}
{-# LANGUAGE LambdaCase            #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE MultiWayIf            #-}
{-# LANGUAGE PatternSynonyms       #-}
{-# LANGUAGE RankNTypes            #-}
{-# LANGUAGE ScopedTypeVariables   #-}
{-# LANGUAGE TypeApplications      #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeInType            #-}
{-# LANGUAGE TypeOperators         #-}
{-# LANGUAGE UndecidableInstances  #-}
{-# LANGUAGE ViewPatterns          #-}

module Model(
  -- * Statistics
    batchMean
  , batchVar
  ) where

import           Control.Category
import           Data.Foldable
import           Data.Profunctor
import           Data.Proxy
import           Data.Type.Tuple
import           GHC.TypeNats
import           Numeric.Backprop hiding                      (Rec(..))
import           Numeric.LinearAlgebra.Static.Backprop hiding (tr)
import           Prelude hiding                               ((.), id)
import qualified Control.Foldl                                as F
import qualified Data.Vector.Sized                            as SV
import qualified Data.Vector.Storable.Sized                   as SVS
import qualified Numeric.LinearAlgebra                        as HU
import qualified Numeric.LinearAlgebra.Static                 as H
import qualified Numeric.LinearAlgebra.Static.Vector          as H

batchMean
    :: (Backprop (t a), Foldable t, Functor t, Fractional a, Reifies s W)
    => BVar s (t a)
    -> BVar s a
batchMean = liftOp1 . op1 $ \xs ->
    let x :& n = F.fold ((:&) <$> F.sum <*> F.length) xs
    in  (x / fromIntegral n, \d -> (d / fromIntegral n) <$ xs)

batchVar
    :: (Backprop (t a), Foldable t, Functor t, Fractional a, Reifies s W)
    => BVar s (t a)
    -> BVar s a
batchVar = liftOp1 . op1 $ \xs ->
    let x2 :& x1 :& x0 = F.fold ((\x2' x1' x0' -> x2' :& x1' :& x0') <$> lmap (^(2::Int)) F.sum <*> F.sum <*> F.length) xs
        meanx  = x1 / fromIntegral x0
        subAll = 2 * x1 / (fromIntegral x0 ^ (2 :: Int))
    in  ( (x2 / fromIntegral x0) - meanx * meanx
        , \d -> let subAllD = d * subAll
                in  (\x -> d * 2 * x / fromIntegral x0 - subAllD) <$> xs
        )
