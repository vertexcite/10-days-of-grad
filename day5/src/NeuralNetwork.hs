-- |= Neural Network Building Blocks

{-# LANGUAGE FlexibleContexts #-}

module NeuralNetwork where
  -- ( NeuralNetwork
  -- , Layer (..)
  -- , Matrix
  -- , Vector
  -- , FActivation (..)
  -- , sigmoid
  -- , sigmoid'
  -- , genWeights
  -- , forward
  --
  -- -- * Training
  -- , sgd
  --
  -- -- * Inference
  -- , accuracy
  -- , avgAccuracy
  -- , inferBinary
  -- , winnerTakesAll
  --
  -- -- * Helpers
  -- , rows
  -- , cols
  -- , computeMap
  -- , rand
  -- , randn
  -- , randomishArray
  -- , scale
  -- , iterN
  -- , mean
  -- , var
  -- , br
  -- ) where

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

