{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}

import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.IO as A
import           Graphics.ColorSpace
import           Text.Printf ( printf )

import           NeuralNetwork
import           Weights

-- TODO: implement algorithm from
-- Accelerating Deep Learning by Focusing on the Biggest Losers

writeImageY :: FilePath -> Matrix Float -> IO ()
writeImageY f a = do
  let b = compute $ A.map pure a :: A.Image U Y Float
  A.writeImageAuto f b

getStencil :: (Int, Int) -> Array U Ix4 Float -> Stencil Ix2 Float Float
getStencil ij w0 = makeCorrelationStencilFromKernel (w ij w0)

zeroPadding k (m', n') = makeLoadArray Seq (Sz2 m' n') 0 $ \ _ -> iforM_ k

w :: (Int, Int) -> Array U Ix4 Float -> Array U Ix2 Float
w (i, j) w0 = compute $ w0 !> i !> j

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

  -- 2D convolution
  let stencils = [getStencil (0, 0) w0, getStencil (1, 0) w0, getStencil (2, 0) w0]
      results = zipWith (,) [0..] $ map (\s -> compute $ mapStencil Edge s im) stencils

  mapM_ (\(i, result) ->  writeImageY (show i ++ ".png") result) results

  -- Equivalent 1D convolution.
  -- See Multidimensional convolution via a 1D convolution algorithm
  -- by Naghizadeh and Sacchi.
  -- We assume that images are in a smaller 20x20 box, so there is no need for
  -- additional padding.
  -- First, prepare the 1D kernel
  let k_ = w (0, 0) w0
      -- Zero padding the kernel to a 28x28 matrix:
      k = compute $ zeroPadding k_ (28, 28) :: Matrix Float
      -- Reshape to a 1D array
      k' = A.resize' 784 k
      -- Crop to a new 1D kernel of 28*4 + 5 = 117 first values
      k2 = compute $ extractFromTo' 0 117 k' :: Vector Float

  -- Second, reshape the image
  let im' = compute $ A.resize' 784 im :: Vector Float

  -- Third, run the 1D convolution and reshape back
  let c = compute $ mapStencil Edge (makeCorrelationStencilFromKernel k2) im' :: Vector Float
      im2 = A.resize' (Sz (28 :. 28)) c
  writeImageY "0_a.png" im2

  return ()
