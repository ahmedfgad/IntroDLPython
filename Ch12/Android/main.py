import kivymd.app
import kivymd.uix.menu
import kivymd.uix.button
import threading
import MLP
import kivy.uix.screenmanager
import pickle

class MainScreen(kivy.uix.screenmanager.Screen):
    pass

class InputFileChooserScreen(kivy.uix.screenmanager.Screen):
    pass

class OutputFileChooserScreen(kivy.uix.screenmanager.Screen):
    pass

class NeuralApp(kivymd.app.MDApp):

    def select_file(self, screen_num, *args):
        path = self.root.screens[screen_num].ids.file_chooser.selection
        if screen_num == 1:
            self.input_file_select_path(path=path)
        else:
            self.output_file_select_path(path=path)

    def on_start(self):
        
#        with open('data_inputs.pkl', 'wb') as out:
#            pickle.dump(numpy.array([[0.1, 0.4, 4.1, 4.3, 1.8, 2.0, 0.01, 0.9, 3.8, 1.6]]), out)

#        with open('data_outputs.pkl', 'wb') as out:
#            pickle.dump(numpy.array([[0.2]]), out)

        self.x = None
        self.y = None
        self.net_arch = None
        self.max_iter = None
        self.tolerance = None
        self.learning_rate = None
        self.activation = None
        self.GD_type = None
        self.debug = False

        self.activation_menu = kivymd.uix.menu.MDDropdownMenu(caller=self.root.screens[0].ids.activation_menu,
                                                              items=[{"text": "sigmoid"}, {"text": "relu"}],
                                                              callback=self.activation_menu_callback,
                                                              width_mult=4)

        self.gdtype_menu = kivymd.uix.menu.MDDropdownMenu(caller=self.root.screens[0].ids.gdtype_menu,
                                                          items=[{"text": "stochastic"}, {"text": "batch"}],
                                                          callback=self.gdtype_menu_callback,
                                                          width_mult=4)

    def activation_menu_callback(self, activation_menu):
        self.root.screens[0].ids.activation_menu.text = activation_menu.text
        self.activation = activation_menu.text
        self.activation_menu.dismiss()

    def gdtype_menu_callback(self, gdtype_menu):
        self.root.screens[0].ids.gdtype_menu.text = gdtype_menu.text
        self.GD_type = gdtype_menu.text
        self.gdtype_menu.dismiss()

    def input_file_select_path(self, path):
        if len(path) == 0:
            self.root.screens[1].ids.select_file_label.text = "Error: No file selected."
            self.root.screens[0].ids.input_data_file_button.text = "Input Data"
            return
        elif path[0][-4:] == ".pkl":
            # kivymd.toast.toast(path)
            with open(path[0], 'rb') as inp:
                self.x = pickle.load(inp)
                # self.x = numpy.load(path[0])
            if self.x.ndim != 2:
                self.root.screens[1].ids.select_file_label.text = "Error: The input array must have 2 dimensions."
                self.x = None
                self.root.screens[0].ids.input_data_file_button.text = "Input Data"
                return
        else:
            self.root.screens[1].ids.select_file_label.text = "Error: A pickle file must be selected."
            self.x = None
            self.root.screens[0].ids.input_data_file_button.text = "Input Data"
            return

        self.root.screens[0].ids.input_data_file_button.text = path[0]
        self.root.current = "main"

    def output_file_select_path(self, path):
        if len(path) == 0:
            self.root.screens[2].ids.select_file_label.text = "Error: No file selected."
            self.root.screens[0].ids.output_data_file_button.text = "Output Data"
            return
        elif path[0][-4:] == ".pkl":
            # kivymd.toast.toast(path)
            with open(path[0], 'rb') as inp:
                self.y = pickle.load(inp)
                # self.y = numpy.load(path[0])
            if self.y.ndim != 2:
                self.root.screens[2].ids.select_file_label.text = "Error: The output array must have 2 dimensions."
                self.y = None
                self.root.screens[0].ids.output_data_file_button.text = "Output Data"
                return
        else:
            self.root.screens[2].ids.select_file_label.text = "Error: A pickle file must be selected."
            self.y = None
            self.root.screens[0].ids.output_data_file_button.text = "Output Data"
            return

        self.root.screens[0].ids.output_data_file_button.text = path[0]
        self.root.current = "main"

    def debug_switch(self, *args):
        self.debug = self.root.screens[0].ids.debug.active

    def button_press(self, *args):        
        self.net_arch = None
        self.max_iter = None
        self.tolerance = None
        self.learning_rate = None

        self.learning_rate = self.root.screens[0].ids.learning_rate.text
        try:
            self.learning_rate = float(self.learning_rate)
            if self.learning_rate >= 0.0 and self.learning_rate <= 1.0:
                self.root.screens[0].ids.label.text = ""
            else:
                self.root.screens[0].ids.label.text = "Wrong value for the learning rate."
                self.learning_rate = None
                return
        except:
            self.root.screens[0].ids.label.text = "Wrong value for the learning rate."
            self.learning_rate = None
            return

        self.tolerance = self.root.screens[0].ids.tolerance.text
        try:
            self.tolerance = float(self.tolerance)
            self.root.screens[0].ids.label.text = ""
        except:
            self.root.screens[0].ids.label.text = "Wrong value for the tolerance."
            self.tolerance = None
            return

        self.max_iter = self.root.screens[0].ids.iterations.text
        try:
            if int(self.max_iter) < 0:
                self.root.screens[0].ids.label.text = "Wrong value for the number of iterations."
                self.max_iter = None
                return
            else:
                self.max_iter = int(self.max_iter)
                self.root.screens[0].ids.label.text = ""
        except:
            self.root.screens[0].ids.label.text = "Wrong value for the number of iterations."
            self.max_iter = None
            return

        net_arch = self.root.screens[0].ids.net_arch.text.split(",")
        temp = []
        if len(net_arch) == 1 and net_arch[0].strip() == '':
            self.net_arch = []
        else:
            for idx in range(len(net_arch)):
                try:
                    if int(net_arch[idx]) <= 0:
                        self.root.screens[0].ids.label.text = "Wrong network architecture."
                        self.net_arch = None
                        return
                    else:
                        temp.append(int(net_arch[idx]))
                except:
                    self.root.screens[0].ids.label.text = "Wrong network architecture."
                    self.net_arch = None
                    return
            self.net_arch = temp

        # all_params = [self.x, self.y, self.net_arch, self.max_iter, self.tolerance, self.learning_rate, self.activation, self.GD_type, self.debug]
        all_params_type = [type(self.x), type(self.y), type(self.net_arch), type(self.max_iter), type(self.tolerance), type(self.learning_rate), type(self.activation), type(self.GD_type), type(self.debug)]
        if type(None) in all_params_type:
            self.root.screens[0].ids.label.text = "Something is wrong. Please check your inputs."
            return

        if self.x.shape[0] != self.y.shape[0]:
            self.root.screens[0].ids.label.text = "Error: Number of samples in the input and output files must match."
            return

        self.root.screens[0].ids.label.text = "All inputs are correct."
        self.root.screens[0].ids.loading.active = True
        self.root.screens[0].ids.btn.disabled = True

        neural_thread = NeuralThread(self, 
                                     self.x, 
                                     self.y, 
                                     self.net_arch, 
                                     self.max_iter, 
                                     self.tolerance, 
                                     self.learning_rate, 
                                     self.activation, 
                                     self.GD_type, 
                                     self.debug)
        neural_thread.start()

class NeuralThread(threading.Thread):

    def __init__(self, app, 
                 x, 
                 y, 
                 net_arch, 
                 max_iter, 
                 tolerance, 
                 learning_rate, 
                 activation, 
                 GD_type,
                 debug):

        super().__init__()

        self.app = app
        self.x = x
        self.y = y
        self.net_arch = net_arch
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.activation = activation
        self.GD_type = GD_type
        self.debug = debug

    def run(self):
        # all_params = [self.x, self.y, self.net_arch, self.max_iter, self.tolerance, self.learning_rate, self.activation, self.GD_type, self.debug]
        self.app.root.screens[0].ids.label.text = "Training started..."

        net = MLP.MLP.train(self.x, 
                            self.y, 
                            self.net_arch, 
                            self.max_iter, 
                            self.tolerance, 
                            self.learning_rate, 
                            self.activation, 
                            self.GD_type, 
                            self.debug)

        self.app.root.screens[0].ids.label.text = "Network is trained. \nTraining time (sec) : {train_time}\nNumber of elapsed iterations : {num_iters}\nNetwork Error : {net_err}".format(train_time=net["training_time_sec"], num_iters=net["elapsed_iter"], net_err=net["network_error"])
        self.app.root.screens[0].ids.loading.active = False
        self.app.root.screens[0].ids.btn.disabled = False

app = NeuralApp()
app.run()

