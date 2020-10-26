from jsbsim_utils import SandBox, create_fdm
import threading
import time
import telnetlib
import os


class JSBSimThread(threading.Thread):
    def __init__(self, fdm, cond, end_time, t0=0.0):
        threading.Thread.__init__(self)
        self.realTime = False
        self.quit = False
        self._fdm = fdm
        self._cond = cond
        self._end_time = end_time
        self._t0 = t0

    def __del_(self):
        del self._fdm

    def run(self):
        self._cond.acquire()
        current_sim_time = self._fdm.get_sim_time()
        self._cond.release()

        while not self.quit:
            if current_sim_time > self._end_time:
                return

            if not self.realTime or current_sim_time < (time.time() - self._t0):
                self._cond.acquire()
                if not self._fdm.run():
                    self._cond.release()
                    return
                self._fdm.check_incremental_hold()
                current_sim_time = self._fdm.get_sim_time()
                self._cond.notify()
                self._cond.release()


class TelnetServer:
    def __init__(self, fdm, end_time, port):
        # Execute JSBSim in a separate thread
        self.cond = threading.Condition()
        self.thread = JSBSimThread(fdm, self.cond, end_time, time.time())
        self.thread.start()

        # Wait for the thread to be started before connecting to telnet session
        self.cond.acquire()
        self.cond.wait()
        try:
            self.tn = telnetlib.Telnet("localhost", port, 2.0)
        finally:
            self.cond.release()

    def __del__(self):
        if 'tn' in self.__dict__.keys():
            self.tn.close()
        self.thread.quit = True
        self.thread.join()
        del self.thread

    def send_command(self, command):
        self.cond.acquire()
        self.tn.write("{}\n".format(command).encode())
        # Wait one time step before reading output
        self.cond.wait()
        msg = self.tn.read_very_eager().decode()
        self.cond.release()
        self.thread.join(0.1)
        return msg

    def get_property_value(self, property):
        msg = self.send_command("get "+property).split('\n')
        return float(msg[0].split('=')[1])

    def set_property_value(self, property):
        self.send_command("set "+property).split('\n')
        return

    def print_info(self):
        print(self.send_command("info"))
        return

    def print_command(self, command):
        print(self.send_command(command))
        return

    def wait(self, seconds):
        self.thread.join(seconds)
        self.thread.join(seconds)

    def set_real_time(self, rt):
        self.thread.realTime = rt


def run_server(script='c1722.xml'):
    sandbox = SandBox()
    # Change working directory to be inside the temp folder
    os.chdir(sandbox())
    fdm = create_fdm(sandbox)
    script_path = sandbox.path_to_jsbsim_file('scripts', script)
    print('Current wd is:', os.getcwd())
    fdm.load_script(os.path.abspath(script_path))
    fdm.run_ic()
    fdm.hold()
    tn = TelnetServer(fdm, 5., 1137)
    return tn
    # tn.print_info()
    # print(tn.get_property_value('inertia/weight-lbs'))
    # print(tn.get_property_value('simulation/sim-time-sec'))
    # tn.print_command('resume')  # runs to end of script
