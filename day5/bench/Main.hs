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
time                 303.9 μs   (295.0 μs .. 313.5 μs)
                     0.997 R²   (0.995 R² .. 0.999 R²)
mean                 299.7 μs   (297.0 μs .. 303.7 μs)
std dev              10.37 μs   (7.740 μs .. 16.03 μs)
variance introduced by outliers: 29% (moderately inflated)

benchmarking Conv2d: no batches (3D)/1 chan -> 3 chan, with padding/Par
time                 540.0 μs   (525.8 μs .. 552.6 μs)
                     0.997 R²   (0.995 R² .. 0.999 R²)
mean                 521.6 μs   (517.0 μs .. 530.0 μs)
std dev              18.81 μs   (14.60 μs .. 25.16 μs)
variance introduced by outliers: 29% (moderately inflated)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Seq
time                 619.4 μs   (615.7 μs .. 626.3 μs)
                     0.998 R²   (0.997 R² .. 1.000 R²)
mean                 634.0 μs   (628.9 μs .. 641.8 μs)
std dev              21.83 μs   (13.64 μs .. 32.63 μs)
variance introduced by outliers: 26% (moderately inflated)

benchmarking Conv2d: no batches (3D)/3 chan -> 3 chan, no padding/Par
time                 567.3 μs   (524.9 μs .. 597.9 μs)
                     0.978 R²   (0.968 R² .. 0.988 R²)
mean                 522.4 μs   (503.8 μs .. 539.9 μs)
std dev              59.63 μs   (49.88 μs .. 77.02 μs)
variance introduced by outliers: 81% (severely inflated)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Seq
time                 1.406 ms   (1.400 ms .. 1.416 ms)
                     1.000 R²   (0.999 R² .. 1.000 R²)
mean                 1.447 ms   (1.439 ms .. 1.457 ms)
std dev              31.20 μs   (25.30 μs .. 38.69 μs)
variance introduced by outliers: 10% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/1 chan -> 3 chan, with padding/Par
time                 1.579 ms   (1.549 ms .. 1.608 ms)
                     0.993 R²   (0.987 R² .. 0.997 R²)
mean                 1.592 ms   (1.568 ms .. 1.629 ms)
std dev              95.80 μs   (66.75 μs .. 128.5 μs)
variance introduced by outliers: 45% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Seq
time                 2.617 ms   (2.552 ms .. 2.709 ms)
                     0.997 R²   (0.994 R² .. 0.999 R²)
mean                 2.711 ms   (2.662 ms .. 2.767 ms)
std dev              171.8 μs   (125.5 μs .. 213.6 μs)
variance introduced by outliers: 45% (moderately inflated)

benchmarking Conv2d: batch size 1 (4D)/3 chan -> 3 chan, no padding/Par
time                 2.885 ms   (2.803 ms .. 2.959 ms)
                     0.994 R²   (0.991 R² .. 0.997 R²)
mean                 2.993 ms   (2.942 ms .. 3.041 ms)
std dev              168.3 μs   (141.4 μs .. 210.0 μs)
variance introduced by outliers: 37% (moderately inflated)
-}
