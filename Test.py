class Car:
    def __init__(self, name=None):
        self.name = name
        self.weels = None

    def print_name(self):
        print(self.name)
        return self

    def add_weels(self, weels):
        print("adding weels")
        self.weels = weels
        return self

    def double_weels(self):
        self.weels = self.weels * 2
        return self


if __name__ == "__main__":
    car = Car("Lambogini")
    car.add_weels(4).double_weels()
    print(car.weels)