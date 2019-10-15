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

maxpoolStencil2x2 :: Stencil Ix2 Float Float
maxpoolStencil2x2 = makeStencil (Sz2 2 2) 0 $ \ get -> let max4 x1 x2 x3 x4 = max (max (max x1 x2) x3) x4 in max4 <$> get 0 <*> get 1 <*> get (0 :. 1) <*> get (1 :. 0)

maxpool2x2 :: Array U Ix2 Float -> Array U Ix2 Float
maxpool2x2 = computeWithStride (Stride 2). mapStencil Edge maxpoolStencil2x2

testA :: Array U Ix2 Float
testA = fromLists' Seq [[1..4],[5..8],[9..12],[13..16]]

-- > testA
-- Array U Seq (Sz (4 :. 4))
--   [ [ 1.0, 2.0, 3.0, 4.0 ]
--   , [ 5.0, 6.0, 7.0, 8.0 ]
--   , [ 9.0, 10.0, 11.0, 12.0 ]
--   , [ 13.0, 14.0, 15.0, 16.0 ]
--   ]
-- > maxpool2x2 testA
-- Array U Seq (Sz (2 :. 2))
--   [ [ 6.0, 8.0 ]
--   , [ 14.0, 16.0 ]
--   ]

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

  -- Layer 1:
  -- 2D convolution
  let stencils = [getStencil (0, 0) w0, getStencil (1, 0) w0, getStencil (2, 0) w0]
      results = map (\s -> compute $ mapStencil Edge s im) stencils :: [Array U Ix2 Float]
      results0 = compute $ A.concat' (Dim 3) $ map (resize' (Sz (1 :> 28 :. 28))) results :: Array U Ix3 Float
      -- ReLU
      featureMaps = compute $ A.map (max 0.0) results0 :: Array U Ix3 Float

  -- TODO: pooling

  -- print $ size featureMaps
  -- mapM_ (\(i, result) ->  writeImageY (show i ++ ".png") result) $ zipWith (,) [0..] results

  -- Layer 2:
  -- 2D convolution over all three channels
  let stencils1 = map (makeCorrelationStencilFromKernel. (w1 !>)) [0..2]
      results1 = map (\s -> compute $ mapStencil Edge s featureMaps) stencils1 :: [Array U Ix3 Float]
      results1' = map (compute. foldrWithin Dim3 (+) 0.0) results1 :: [Array U Ix2 Float]  -- Reduce the last dimension
      results10 = compute $ A.concat' (Dim 3) $ map (resize' (Sz (1 :> 28 :. 28))) results1' :: Array U Ix3 Float
      featureMaps1 = compute $ A.map (max 0.0) results10 :: Array U Ix3 Float

  print $ size featureMaps1
  -- Sz (3 :> 28 :. 28)

  print featureMaps1

  -- -- Equivalent 1D convolution.
  -- -- See Multidimensional convolution via a 1D convolution algorithm
  -- -- by Naghizadeh and Sacchi.
  -- -- We assume that images are in a smaller 20x20 box, so there is no need for
  -- -- additional padding.
  -- -- First, prepare the 1D kernel
  -- let k_ = w (0, 0) w0
  --     -- Zero padding the kernel to a 28x28 matrix:
  --     k = compute $ zeroPadding k_ (28, 28) :: Matrix Float
  --     -- Reshape to a 1D array
  --     k' = A.resize' 784 k
  --     -- Crop to a new 1D kernel of 28*4 + 5 = 117 first values
  --     k2 = compute $ extractFromTo' 0 117 k' :: Vector Float
  --
  -- -- Second, reshape the image
  -- let im' = compute $ A.resize' 784 im :: Vector Float
  --
  -- -- Third, run the 1D convolution and reshape back
  -- let c = compute $ mapStencil Edge (makeCorrelationStencilFromKernel k2) im' :: Vector Float
  --     im2 = A.resize' (Sz (28 :. 28)) c
  -- writeImageY "0_a.png" im2

  return ()

-- > a = computeAs U $ resize' (Sz2 5 5) (Ix1 11 ... 35)
-- > d = makeArrayR D Seq (Sz2 10 10) (const 0)
-- > insertWindow d (Window (2 :. 3) (size a) ((a !) . liftIndex2 subtract (2 :. 3)) Nothing)
-- Array DW Seq (Sz (10 :. 10))
--   [ [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 11, 12, 13, 14, 15, 0, 0 ]
--   , [ 0, 0, 0, 16, 17, 18, 19, 20, 0, 0 ]
--   , [ 0, 0, 0, 21, 22, 23, 24, 25, 0, 0 ]
--   , [ 0, 0, 0, 26, 27, 28, 29, 30, 0, 0 ]
--   , [ 0, 0, 0, 31, 32, 33, 34, 35, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   , [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
--   ]
