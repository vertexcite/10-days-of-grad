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
        "Conv2d Seq"
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' padding w0))
            , env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr padding w0))
            , env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "No lists at all".
                 whnf (computeAs U. conv2d_ padding w0))
            , env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "List based".
                 whnf (computeAs U. conv2d' padding w0))
            ]
        ]

    , bgroup
        "Conv2d Seq"
        [ bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm2))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' noPadding w1))
            , env
                (return (setComp Seq $ computeAs U fm2))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr noPadding w1))
            , env
                (return (setComp Seq $ computeAs U fm2))
                (bench "No lists at all".
                 whnf (computeAs U. conv2d_ noPadding w1))
            , env
                (return (setComp Seq $ computeAs U fm2))
                (bench "List based".
                 whnf (computeAs U. conv2d' noPadding w1))
            ]
        ]

    , bgroup
        "Conv2d Par (6 cores)"
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' padding w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr padding w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "No lists at all".
                 whnf (computeAs U. conv2d_ padding w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "List based".
                 whnf (computeAs U. conv2d' padding w0))
            ]
        ]
    , bgroup
        "Conv2d Par (6 cores)"
        [ bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Par $ computeAs U fm2))
                (bench "Delayed array based, foldl'".
                 whnf (computeAs U. conv2dl' noPadding w1))
            , env
                (return (setComp Par $ computeAs U fm2))
                (bench "Delayed array based, foldr".
                 whnf (computeAs U. conv2dr noPadding w1))
            , env
                (return (setComp Par $ computeAs U fm2))
                (bench "No lists at all".
                 whnf (computeAs U. conv2d_ noPadding w1))
            , env
                (return (setComp Par $ computeAs U fm2))
                (bench "List based".
                 whnf (computeAs U. conv2d' noPadding w1))
            ]
        ]
    ]


{-
benchmarking Conv2d Seq/1 chan -> 3 chan, with padding/Delayed array based, foldl'
time                 289.8 μs   (289.0 μs .. 291.8 μs)
                     0.999 R²   (0.997 R² .. 1.000 R²)
mean                 289.9 μs   (288.5 μs .. 295.2 μs)
std dev              8.687 μs   (912.6 ns .. 18.43 μs)
variance introduced by outliers: 24% (moderately inflated)

benchmarking Conv2d Seq/1 chan -> 3 chan, with padding/Delayed array based, foldr
time                 293.3 μs   (291.6 μs .. 295.5 μs)
                     1.000 R²   (0.999 R² .. 1.000 R²)
mean                 290.9 μs   (289.8 μs .. 291.9 μs)
std dev              3.348 μs   (2.569 μs .. 4.992 μs)

benchmarking Conv2d Seq/1 chan -> 3 chan, with padding/No lists at all
time                 411.3 μs   (409.4 μs .. 413.6 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 411.6 μs   (411.2 μs .. 412.3 μs)
std dev              1.518 μs   (805.8 ns .. 2.791 μs)

benchmarking Conv2d Seq/1 chan -> 3 chan, with padding/List based
time                 420.6 μs   (414.6 μs .. 426.7 μs)
                     0.998 R²   (0.996 R² .. 0.999 R²)
mean                 417.3 μs   (414.3 μs .. 427.5 μs)
std dev              17.77 μs   (8.260 μs .. 36.16 μs)
variance introduced by outliers: 37% (moderately inflated)


benchmarking Conv2d Seq/3 chan -> 3 chan, no padding/Delayed array based, foldl'
time                 504.6 μs   (502.2 μs .. 506.5 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 500.3 μs   (499.1 μs .. 501.7 μs)
std dev              4.334 μs   (3.556 μs .. 4.970 μs)

benchmarking Conv2d Seq/3 chan -> 3 chan, no padding/Delayed array based, foldr
time                 497.7 μs   (497.0 μs .. 498.6 μs)
                     1.000 R²   (0.999 R² .. 1.000 R²)
mean                 501.5 μs   (499.2 μs .. 506.8 μs)
std dev              10.14 μs   (3.109 μs .. 18.16 μs)
variance introduced by outliers: 11% (moderately inflated)

benchmarking Conv2d Seq/3 chan -> 3 chan, no padding/No lists at all
time                 573.1 μs   (571.8 μs .. 574.8 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 572.4 μs   (571.8 μs .. 573.6 μs)
std dev              2.769 μs   (1.866 μs .. 4.461 μs)

benchmarking Conv2d Seq/3 chan -> 3 chan, no padding/List based
time                 569.5 μs   (568.6 μs .. 570.2 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 570.6 μs   (570.0 μs .. 571.2 μs)
std dev              1.971 μs   (1.709 μs .. 2.402 μs)


benchmarking Conv2d Par (6 cores)/1 chan -> 3 chan, with padding/Delayed array based, foldl'
time                 739.4 μs   (693.2 μs .. 777.9 μs)
                     0.974 R²   (0.961 R² .. 0.984 R²)
mean                 674.4 μs   (646.6 μs .. 702.6 μs)
std dev              89.01 μs   (75.50 μs .. 106.8 μs)
variance introduced by outliers: 84% (severely inflated)


benchmarking Conv2d Par (6 cores)/1 chan -> 3 chan, with padding/Delayed array based, foldr
time                 886.2 μs   (814.1 μs .. 976.4 μs)
                     0.919 R²   (0.888 R² .. 0.950 R²)
mean                 866.0 μs   (800.5 μs .. 935.4 μs)
std dev              215.5 μs   (181.6 μs .. 250.2 μs)
variance introduced by outliers: 96% (severely inflated)

benchmarking Conv2d Par (6 cores)/1 chan -> 3 chan, with padding/No lists at all
time                 827.8 μs   (750.6 μs .. 908.4 μs)
                     0.959 R²   (0.938 R² .. 0.985 R²)
mean                 861.5 μs   (833.3 μs .. 902.6 μs)
std dev              123.5 μs   (101.9 μs .. 157.3 μs)
variance introduced by outliers: 86% (severely inflated)

benchmarking Conv2d Par (6 cores)/1 chan -> 3 chan, with padding/List based
time                 920.5 μs   (874.9 μs .. 988.1 μs)
                     0.972 R²   (0.958 R² .. 0.984 R²)
mean                 967.4 μs   (932.3 μs .. 1.009 ms)
std dev              126.3 μs   (111.3 μs .. 148.0 μs)
variance introduced by outliers: 82% (severely inflated)


benchmarking Conv2d Par (6 cores)/3 chan -> 3 chan, no padding/Delayed array based, foldl'
time                 855.3 μs   (828.8 μs .. 893.9 μs)
                     0.983 R²   (0.969 R² .. 0.994 R²)
mean                 873.2 μs   (846.2 μs .. 909.8 μs)
std dev              114.8 μs   (69.39 μs .. 165.1 μs)
variance introduced by outliers: 83% (severely inflated)

benchmarking Conv2d Par (6 cores)/3 chan -> 3 chan, no padding/Delayed array based, foldr
time                 876.7 μs   (829.8 μs .. 930.9 μs)
                     0.977 R²   (0.964 R² .. 0.995 R²)
mean                 837.2 μs   (807.5 μs .. 883.0 μs)
std dev              126.5 μs   (86.85 μs .. 175.8 μs)
variance introduced by outliers: 87% (severely inflated)

benchmarking Conv2d Par (6 cores)/3 chan -> 3 chan, no padding/No lists at all
time                 999.8 μs   (937.9 μs .. 1.062 ms)
                     0.969 R²   (0.955 R² .. 0.981 R²)
mean                 1.020 ms   (973.7 μs .. 1.074 ms)
std dev              172.3 μs   (140.8 μs .. 227.0 μs)
variance introduced by outliers: 90% (severely inflated)

benchmarking Conv2d Par (6 cores)/3 chan -> 3 chan, no padding/List based
time                 896.4 μs   (845.3 μs .. 964.2 μs)
                     0.955 R²   (0.924 R² .. 0.986 R²)
mean                 904.1 μs   (871.6 μs .. 960.1 μs)
std dev              152.6 μs   (116.2 μs .. 198.7 μs)
variance introduced by outliers: 90% (severely inflated)
-}
