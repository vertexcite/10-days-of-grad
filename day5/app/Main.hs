{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeFamilies #-}

import           Data.Massiv.Array hiding ( map, zip, unzip )
import qualified Data.Massiv.Array as A
import           Text.Printf ( printf )

import           NeuralNetwork
import           Weights

-- TODO: implement algorithm from
-- Accelerating Deep Learning by Focusing on the Biggest Losers

main :: IO ()
main = do
  let a = A.fromList Seq [1,2,5,6,2,-1,3,-2] :: Vector Int
  print a

  -- 1D identity kernel
  let s = makeCorrelationStencilFromKernel (A.fromList Seq [0, 1, 0] :: Vector Int)
  let delayedS = makeCorrelationStencilFromKernel (A.fromList Seq [1, 0, 0] :: Vector Int)
  print $ mapStencil Edge s a
  print (compute $ mapStencil Edge delayedS a :: Vector Int)
  let stride = Stride 3
  print (computeWithStride stride $ mapStencil Edge delayedS a :: Vector Int)
  return ()
