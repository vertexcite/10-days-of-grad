{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DataKinds #-}

import           Data.Massiv.Array hiding ( map, zip, unzip, zipWith, mapM_ )
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as A
-- import qualified Data.Massiv.Array.IO as A
-- import           Graphics.ColorSpace
import           Streamly
import qualified Streamly.Prelude as S
import           Text.Printf ( printf )
import           Control.DeepSeq ( force )
import           Control.Monad.Trans.Maybe
import           Data.IDX
import qualified Data.Vector.Unboxed as V
import           Data.List.Split ( chunksOf )

import           NeuralNetwork
import           Shuffle ( shuffleIO )
-- import           Weights

-- TODO: implement algorithm from
-- Accelerating Deep Learning by Focusing on the Biggest Losers

loadMNIST
  :: FilePath -> FilePath -> IO (Maybe [(Volume4 Float, Matrix Float)])
loadMNIST fpI fpL = runMaybeT $ do
    i <- MaybeT $ decodeIDXFile fpI
    l <- MaybeT $ decodeIDXLabelsFile fpL
    d <- MaybeT. return $ force $ labeledIntData l i
    return $ map _conv d
  where
    _conv :: (Int, V.Vector Int) -> (Volume4 Float, Matrix Float)
    _conv (label, v) = (v1, toOneHot10 label)
      where
        v0 = V.map ((/0.3081). (`subtract` 0.1307). (/ 255). fromIntegral) v

        v1 = A.fromVector' Par (Sz4 1 1 28 28) v0

toOneHot10 :: Int -> Matrix Float
toOneHot10 n = A.makeArrayR U Par (Sz2 1 10) (\(_ :. j) -> if j == n then 1 else 0)

mnistStream
  :: Int -> FilePath -> FilePath
  -> IO (SerialT IO (Volume4 Float, Matrix Float))
mnistStream batchSize fpI fpL = do
  Just dta <- loadMNIST fpI fpL
  dta2 <- shuffleIO dta

  -- Split data into batches
  let (vs, labs) = unzip dta2
      merge4 :: [Volume4 Float] -> Volume4 Float
      -- Dimensions:
      -- Sz (1 :> 1 :> 1 :. 1)
      --     ^    ^    ^    ^
      --     4    |    |    1
      --          3    2
      -- So the first dimension is actually
      -- fourth in this notation
      merge4 = setComp Par. A.compute. A.concat' 4

      merge2 :: [Matrix Float] -> Matrix Float
      merge2 = setComp Par. A.compute. A.concat' 2

      vs' = map merge4 $ chunksOf batchSize vs
      labs' = map merge2 $ chunksOf batchSize labs
      dta' = zip vs' labs'
  return $ S.fromList dta'

data TrainSettings = TrainSettings
  { _printEpochs :: Int  -- Print every N epochs
  , _lr :: Float  -- Learning rate
  , _totalEpochs :: Int  -- Number of training epochs
  }

train
  :: TrainSettings
  -> BNN Float
  -> (SerialT IO (Volume4 Float, Matrix Float),
      SerialT IO (Volume4 Float, Matrix Float))
  -> IO (BNN Float)
train TrainSettings { _printEpochs = printEpochs
                    , _lr = lr
                    , _totalEpochs = totalEpochs
                    } net (trainS, testS) = do
  (net', _) <- iterN (totalEpochs `div` printEpochs) (\(net0, j) -> do
    net1 <- sgd lr printEpochs net0 trainS

    tacc <- net1 `avgAccuracy` trainS :: IO Float
    putStr $ printf "%d Training accuracy %.1f" (j :: Int) tacc

    acc <- net1 `avgAccuracy` testS :: IO Float
    putStrLn $ printf "  Validation accuracy %.1f" acc

    return (net1, j + printEpochs)
    ) (net, 1)
  return net'

main :: IO ()
main = do
  trainS <- mnistStream 64 "data/train-images-idx3-ubyte" "data/train-labels-idx1-ubyte"
  testS <- mnistStream 1000 "data/t10k-images-idx3-ubyte" "data/t10k-labels-idx1-ubyte"

  net <- randNetwork

  net' <- train TrainSettings { _printEpochs = 1
                              , _lr = 0.01
                              , _totalEpochs = 30
                              } net (trainS, testS)

  return ()
