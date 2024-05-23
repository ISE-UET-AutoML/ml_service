import ultralytics
import transformers
if __name__ == "__main__":
    model = ultralytics.YOLO()

    
    results = model("/home/xuananle/Documents/AutoML-Platform/src/mlservice/app/test/bus.jpg")

    # results[0].show()

    results[0].save("output.jpg")



