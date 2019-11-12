import           Criterion.Main
import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as A
import           Data.List ( foldl' )

import           Weights ( im, w0, w1 )


-- | 2D convolution
conv2d'
       :: Padding Ix3 Float  -- ^ Padding
       -> Array U Ix4 Float  -- ^ Weights
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2d' padding w x = compute $ A.concat' (Dim 3) results
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    stencils = map (makeCorrelationStencilFromKernel. (w !>)) [0..cout - 1]
    results :: [Array U Ix3 Float]
    results = map (\s -> compute $ applyStencil padding s x) stencils

conv2d_
       :: Padding Ix3 Float  -- ^ Padding
       -> Array U Ix4 Float  -- ^ Weights
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2d_ padding w x = compute $ A.concat' (Dim 3) results
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    stencils = makeCorrelationStencilFromKernel. computeAs U. (w !>) <$> (0 ..: cout)
    results :: Array D Ix1 (Array U Ix3 Float)
    results = fmap (\s -> compute $ applyStencil padding s x) stencils

conv2dr
       :: Padding Ix3 Float  -- ^ Padding
       -> Array U Ix4 Float  -- ^ Weights
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2dr padding w x = compute res
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    base = computeAs U $ applyStencil padding (makeCorrelationStencilFromKernel $ w !> 0) x
    res = foldr (\ch prev -> let stencil = makeCorrelationStencilFromKernel $ w !> ch
                                 conv = computeAs U $ applyStencil padding stencil x
                             in computeAs U $ append' 3 prev conv) base [1..cout - 1]

conv2dl'
       :: Padding Ix3 Float  -- ^ Padding
       -> Array U Ix4 Float  -- ^ Weights
       -> Array U Ix3 Float  -- ^ Input features
       -> Array U Ix3 Float  -- ^ Output features
conv2dl' padding w x = compute res
  where
    (Sz (cout :> cin :> _ :. _)) = size w
    base = computeAs U $ applyStencil padding (makeCorrelationStencilFromKernel $ w !> 0) x
    res = foldl' (\prev ch -> let stencil = makeCorrelationStencilFromKernel $ w !> ch
                                  conv = computeAs U $ applyStencil padding stencil x
                              in computeAs U $ append' 3 prev conv) base [1..cout - 1]

main :: IO ()
main = do
  let padding = Padding (Sz3 0 2 2) (Sz3 0 2 2) (Fill 0.0) :: Padding Ix3 Float
      fm2 = computeAs U. conv2dl' padding w0. resize' (Sz (1 :> 28 :. 28)) $ im
  defaultMain
    [ bgroup
        "Conv2d"
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' padding w0))
            , env
                (return (resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr padding w0))
            , env
                (return (resize' (Sz (1 :> 28 :. 28)) im))
                (bench "No lists at all".
                 whnf (computeAs U. conv2d_ padding w0))
            , env
                (return (resize' (Sz (1 :> 28 :. 28)) im))
                (bench "List based".
                 whnf (computeAs U. conv2d' padding w0))
            ]
        ]
    , bgroup
        "Conv2d"
        [ bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (computeAs U fm2))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' noPadding w1))
            , env
                (return (computeAs U fm2))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr noPadding w1))
            , env
                (return (computeAs U fm2))
                (bench "List based".
                 whnf (computeAs U. conv2d' noPadding w1))
            ]
        ]
    ]

-- benchmarking Conv2d/1 chan -> 3 chan, with padding/Delayed array based, foldl'
-- time                 345.3 μs   (344.2 μs .. 347.6 μs)
--                      1.000 R²   (0.999 R² .. 1.000 R²)
-- mean                 345.9 μs   (345.3 μs .. 348.5 μs)
-- std dev              3.836 μs   (1.283 μs .. 7.766 μs)
--
-- benchmarking Conv2d/1 chan -> 3 chan, with padding/Delayed array based, foldr
-- time                 346.2 μs   (345.7 μs .. 346.7 μs)
--                      1.000 R²   (1.000 R² .. 1.000 R²)
-- mean                 346.7 μs   (346.4 μs .. 347.3 μs)
-- std dev              1.390 μs   (1.026 μs .. 1.841 μs)
--
-- benchmarking Conv2d/1 chan -> 3 chan, with padding/List based
-- time                 388.4 μs   (387.0 μs .. 390.2 μs)
--                      1.000 R²   (0.999 R² .. 1.000 R²)
-- mean                 387.3 μs   (386.3 μs .. 389.7 μs)
-- std dev              4.971 μs   (2.397 μs .. 9.977 μs)
--
-- benchmarking Conv2d/3 chan -> 3 chan, no padding/Delayed array based, foldl'
-- time                 482.7 μs   (481.8 μs .. 483.3 μs)
--                      1.000 R²   (1.000 R² .. 1.000 R²)
-- mean                 483.8 μs   (483.3 μs .. 484.4 μs)
-- std dev              1.845 μs   (1.384 μs .. 2.553 μs)
--
-- benchmarking Conv2d/3 chan -> 3 chan, no padding/Delayed array based, foldr
-- time                 483.4 μs   (481.9 μs .. 485.0 μs)
--                      1.000 R²   (1.000 R² .. 1.000 R²)
-- mean                 483.6 μs   (482.8 μs .. 484.6 μs)
-- std dev              2.803 μs   (2.067 μs .. 4.001 μs)
--
-- benchmarking Conv2d/3 chan -> 3 chan, no padding/List based
-- time                 539.0 μs   (537.8 μs .. 540.0 μs)
--                      1.000 R²   (1.000 R² .. 1.000 R²)
-- mean                 540.0 μs   (539.2 μs .. 541.1 μs)
-- std dev              3.031 μs   (2.308 μs .. 4.195 μs)
