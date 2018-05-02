import multiprocessing
from lxyTools.progressBar import simpleBar
import time


def processRun(obj, i):
    for val in obj.processPeriodlist[i]:
        # print(obj.progressCount.value)
        obj.progressCount[i] += 1
        re = obj.fun(val)

    obj.result.put(re)


def progressBarUpdate(obj):
    while 1:
        val = 0
        for i in obj.progressCount:
            val += i

        obj.bar.update(val)
        time.sleep(0.5)
        if val >= len(obj.period):
            obj.bar.finish()
            break


class mutiProcessLoop:
    def __init__(self, fun, period, n_process=4, silence=False):
        self.period = period
        self.n_process = n_process
        self.silence = silence
        self.fun = fun
        self.processPeriodlist = []
        self.processlist = []
        self.manager = multiprocessing.Manager()
        self.result = multiprocessing.Queue(n_process)
        self.progressCount = self.manager.Array('i', [0]*n_process)
        for i in range(n_process):
            self.processPeriodlist.append([])
            self.processlist.append(
                multiprocessing.Process(
                    target=processRun,
                    args=(self, i))
                )
        for i in range(len(period)):
            self.processPeriodlist[i % n_process].append(period[i])

        if not silence:
            self.bar = simpleBar(len(period))
            self.processlist.append(multiprocessing.Process(
                target=progressBarUpdate,
                args=(self,)))

    def run(self):
        for process in self.processlist:
            process.start()

        for process in self.processlist:
            process.join()
            process.terminate()

        resultlist = []

        for i in range(self.n_process):
            resultlist.append(self.result.get())

        return resultlist
