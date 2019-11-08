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
maxpool2x2 = computeWithStride (Stride 2). applyStencil noPadding maxpoolStencil2x2

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

relu :: Array U Ix3 Float -> Array U Ix3 Float
relu = compute. A.map (max 0.0)

-- | 2D convolution
conv2d :: Array U Ix4 Float  -- ^ Weights
       -> Padding Ix3 Float  -- ^ Padding
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2d w padding x = compute $ A.concat' (Dim 3) results
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    stencils = map (makeCorrelationStencilFromKernel. (w !>)) [0..cout - 1]
    results :: [Array U Ix3 Float]
    results = map (\s -> compute $ applyStencil padding s x) stencils

infixl 9 ~>
(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f

testLeNet :: IO ()
testLeNet = do
  -- By convention, the first dimension is channels
  let im1channel = resize' (Sz (1 :> 28 :. 28)) im
      lenetFeatures = conv2d w0 (Padding (Sz3 0 2 2) (Sz3 0 2 2) (Fill 0.0))
                    ~> relu
                    ~> conv2d w1 noPadding
                    ~> relu
      featureMaps2 = lenetFeatures im1channel

  print $ size featureMaps2
  -- Sz (3 :> 24 :. 24)

  -- mapM_ (\(i, result) ->  writeImageY (show i ++ ".png") result) $ zipWith (,) [0..]  (splitChannels featureMaps2)

testConv1D :: IO ()
testConv1D = do
  let a = A.fromList Seq [1,2,5,6,2,-1,3,-2] :: Vector Int
  print a

  -- 1D identity kernel
  let s = makeCorrelationStencilFromKernel (A.fromList Seq [0, 1, 0] :: Vector Int)
  let delayedS = makeCorrelationStencilFromKernel (A.fromList Seq [1, 0, 0] :: Vector Int)
  print $ mapStencil (Fill 0) s a
  print (compute $ mapStencil (Fill 0) delayedS a :: Vector Int)
  let stride = Stride 3
  print (computeWithStride stride $ mapStencil Edge delayedS a :: Vector Int)

main :: IO ()
main = putStrLn "Test 1D convolutions" >> testConv1D
       >> putStrLn "Test 2D convolutions" >> testLeNet

-- zeroPadding (m', n') = compute. applyStencil (Padding (Sz2 0 0) (Sz2 m' n') (Fill 0)) idStencil
--
-- test2Dto1D = do
--   -- 1D convolution equivalent to 2D:
--   -- (1) Append zeros to make both kernel and image square matrices.
--   -- (2) Flatten both.
--   -- (3) Reject zero tails.
--   -- See Multidimensional convolution via a 1D convolution algorithm
--   -- by Naghizadeh and Sacchi.
--
--   -- Here we assume that images are in a smaller 20x20 box, so there is no need for
--   -- additional padding.
--   -- First, prepare the 1D kernel
--   let k_ = w (0, 0) w0
--       -- Zero padding the kernel to a 28x28 matrix:
--       k = zeroPadding (23, 23) k_ :: Matrix Float
--       -- Reshape to a 1D array
--       k' = A.resize' 784 k
--       -- Crop to a new 1D kernel of 28*4 + 5 = 117 first values
--       k2 = compute $ extractFromTo' 0 117 k' :: Vector Float
--
--   -- Second, reshape the image
--   let im' = compute $ A.resize' 784 im :: Vector Float
--
--   -- Third, run the 1D convolution and reshape back
--   let c = compute $ mapStencil Edge (makeCorrelationStencilFromKernel k2) im' :: Vector Float
--       im2 = A.resize' (Sz (28 :. 28)) c
--   writeImageY "0_a.png" im2
--
--   return ()

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
