import           Criterion.Main
import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as A
import           Data.List ( foldl' )

import           Weights ( im, w0, w1 )
import           NeuralNetwork ( conv2d_ )


-- | 2D convolution on individual image
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
  let pad2 = Padding (Sz2 2 2) (Sz2 2 2) (Fill 0.0) :: Padding Ix2 Float
      nopad2 = Padding (Sz2 0 0) (Sz2 0 0) (Fill 0.0) :: Padding Ix2 Float
      pad3 = Padding (Sz3 0 2 2) (Sz3 0 2 2) (Fill 0.0) :: Padding Ix3 Float
      nopad3 = Padding (Sz3 0 0 0) (Sz3 0 0 0) (Fill 0.0) :: Padding Ix3 Float
      fm' = computeAs U. conv2d_ pad2 w0. resize' (Sz (1 :> 1 :> 28 :. 28)) $ im
      fm0 = computeAs U. conv2dl' pad3 w0. resize' (Sz (1 :> 28 :. 28)) $ im
  defaultMain
    [ bgroup
        "Conv2d: no batches (3D)"  -- TODO: increase batch size to see if parallelization is useful
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Seq".
                 whnf (computeAs U. conv2dl' pad3 w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 28 :. 28)) im))
                (bench "Par".
                 whnf (computeAs U. conv2dl' pad3 w0))
            ]
        , bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm0))
                (bench "Seq".
                 whnf (computeAs U. conv2dl' nopad3 w0))
            , env
                (return (setComp Par $ computeAs U fm0))
                (bench "Par".
                 whnf (computeAs U. conv2dl' nopad3 w0))
            ]
        ]

    , bgroup
        "Conv2d: batch size 1 (4D)"  -- TODO: increase batch size to see if parallelization is useful
        [ bgroup
            "1 chan -> 3 chan, with padding"
            [ env
                (return (setComp Seq $ resize' (Sz (1 :> 1 :> 28 :. 28)) im))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ pad2 w0))
            , env
                (return (setComp Par $ resize' (Sz (1 :> 1 :> 28 :. 28)) im))
                (bench "Par".
                 whnf (computeAs U. conv2d_ pad2 w0))
            ]
        , bgroup
            "3 chan -> 3 chan, no padding"
            [ env
                (return (setComp Seq $ computeAs U fm'))
                (bench "Seq".
                 whnf (computeAs U. conv2d_ nopad2 w0))
            , env
                (return (setComp Par $ computeAs U fm'))
                (bench "Par".
                 whnf (computeAs U. conv2d_ nopad2 w0))
            ]
        ]

    ]

{-

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Seq
time                 312.5 μs   (310.7 μs .. 315.1 μs)
                     0.999 R²   (0.999 R² .. 1.000 R²)
mean                 321.6 μs   (318.5 μs .. 326.8 μs)
std dev              13.23 μs   (8.305 μs .. 22.30 μs)
variance introduced by outliers: 37% (moderately inflated)

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Par
time                 523.9 μs   (513.0 μs .. 537.9 μs)
                     0.995 R²   (0.991 R² .. 0.999 R²)
mean                 518.2 μs   (511.9 μs .. 525.7 μs)
std dev              23.89 μs   (17.99 μs .. 34.36 μs)
variance introduced by outliers: 40% (moderately inflated)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Seq
time                 634.9 μs   (631.3 μs .. 640.7 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 651.0 μs   (648.0 μs .. 655.6 μs)
std dev              12.38 μs   (9.709 μs .. 16.76 μs)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Par
time                 594.7 μs   (570.1 μs .. 616.9 μs)
                     0.982 R²   (0.971 R² .. 0.989 R²)
mean                 534.3 μs   (514.7 μs .. 558.1 μs)
std dev              73.43 μs   (64.96 μs .. 84.56 μs)
variance introduced by outliers: 86% (severely inflated)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Seq
time                 692.2 μs   (689.7 μs .. 695.3 μs)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 703.6 μs   (701.7 μs .. 705.2 μs)
std dev              6.035 μs   (5.080 μs .. 7.719 μs)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Par
time                 989.3 μs   (973.4 μs .. 1.005 ms)
                     0.997 R²   (0.995 R² .. 0.998 R²)
mean                 942.8 μs   (929.6 μs .. 962.6 μs)
std dev              54.01 μs   (47.89 μs .. 61.30 μs)
variance introduced by outliers: 46% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Seq
time                 1.248 ms   (1.245 ms .. 1.252 ms)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 1.267 ms   (1.262 ms .. 1.285 ms)
std dev              36.49 μs   (6.083 μs .. 70.13 μs)
variance introduced by outliers: 17% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Par
time                 1.705 ms   (1.667 ms .. 1.747 ms)
                     0.996 R²   (0.993 R² .. 0.998 R²)
mean                 1.617 ms   (1.595 ms .. 1.641 ms)
std dev              74.28 μs   (61.64 μs .. 102.5 μs)
variance introduced by outliers: 33% (moderately inflated)
-}
