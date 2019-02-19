{-# LANGUAGE DataKinds                            #-}
{-# LANGUAGE FlexibleContexts                     #-}
{-# LANGUAGE FlexibleInstances                    #-}
{-# LANGUAGE MultiParamTypeClasses                #-}
{-# LANGUAGE PartialTypeSignatures                #-}
{-# LANGUAGE TypeOperators                        #-}
{-# LANGUAGE TypeSynonymInstances                 #-}
{-# LANGUAGE UndecidableInstances                 #-}
{-# OPTIONS_GHC -fno-warn-orphans                 #-}
{-# OPTIONS_GHC -fno-warn-partial-type-signatures #-}

import           Model ( batchMean, batchVar )
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
import           Data.Primitive.MutVar
import           Data.Time
import           Data.Traversable
import           Data.Tuple
import           GHC.TypeNats
import           Numeric.LinearAlgebra.Static.Backprop
import           Numeric.Opto
import           System.Environment
import           System.FilePath
import           Text.Printf
import qualified Data.Conduit.Combinators              as C
import qualified Data.Vector.Generic                   as VG
import qualified Numeric.LinearAlgebra                 as HM
import qualified Numeric.LinearAlgebra.Static          as H
import qualified System.Random.MWC                     as MWC

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

main = do
  let res = evalBP runBatchNorm batch
  mapM_ (print. H.extract) res
