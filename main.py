from tkinter import *
from tkinter import Button, Entry, Label, LabelFrame, Radiobutton, Spinbox, StringVar, Tk, Toplevel, filedialog
from tkinter.scrolledtext import ScrolledText

import platform

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import Lambda, Model, Polynom, Weight

np.set_printoptions(linewidth=np.inf)
plt.style.use("classic")

window = Tk()
window.title("Бригада № 6")
window.resizable(True, True)
window.geometry("1000x800")
window.configure(bg="#FECDA6")
if (platform.system() == 'Linux'):
    window.attributes('-zoomed', True)
elif (platform.system() == 'Darwin'):
    window.attributes('-fullscreen', True)
FOREGROUND = "#ECE3CE"


class Application:
    def __init__(self, window):
        self.samples_label_frame = LabelFrame(window, text="Вхідні дані", bg=FOREGROUND)
        self.samples_label_frame.grid(row=0, column=3, sticky="NW", padx=5, pady=5, ipadx=5, ipady=5)

        self.dimensions_label_frame = LabelFrame(window, text="Розмірності векторів", bg=FOREGROUND)
        self.dimensions_label_frame.grid(row=1, column=3, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_label_frame = LabelFrame(window, text="Поліноми", bg=FOREGROUND)
        self.polynomials_label_frame.grid(row=3, column=3, rowspan=2, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_type_label_frame = LabelFrame(self.polynomials_label_frame, text="Вид поліномів", bg=FOREGROUND)
        self.polynomials_type_label_frame.grid(row=0, column=0, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.polynomials_dimensions_label_frame = LabelFrame(self.polynomials_label_frame, text="Степені поліномів", bg=FOREGROUND)
        self.polynomials_dimensions_label_frame.grid(row=1, column=0, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.additional_label_frame = LabelFrame(window, text="Додатково", bg=FOREGROUND)
        self.additional_label_frame.grid(row=5, column=3, rowspan=2, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.weight_label_frame = LabelFrame(self.additional_label_frame, text="Ваги цільових функцій", bg=FOREGROUND)
        self.weight_label_frame.grid(row=0, column=0, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.lambdas_label_frame = LabelFrame(self.additional_label_frame, text="Метод визначення лямбд", bg=FOREGROUND)
        self.lambdas_label_frame.grid(row=1, column=0, sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

        self.results_label_frame = LabelFrame(window, text="Результати", bg=FOREGROUND)
        self.results_label_frame.grid(row=0, column=0, rowspan=7, columnspan=3, sticky="NS", padx=5, pady=5, ipadx=5, ipady=5)

        # 'Вхідні дані'
        self.samples_label = Label(self.samples_label_frame, text="Файл з вибіркою:")
        self.samples_label.grid(row=0, column=0, sticky="E", padx=5, pady=2)

        self.samples_filename_var = StringVar()
        self.samples_filename_var.set("")
        self.samples_filename_entry = Entry(
            self.samples_label_frame,
            textvariable=self.samples_filename_var,
            state=DISABLED,
        )
        self.samples_filename_entry.grid(row=0, column=1, sticky="WE", padx=5, pady=2)

        self.browse_button = Button(self.samples_label_frame, text="...", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, sticky="W", padx=5, pady=2)

        self.output_filename_label = Label(self.samples_label_frame, text="Вихідний файл:")
        self.output_filename_label.grid(row=1, column=0, sticky="E", padx=5, pady=2)

        self.output_filename_var = StringVar()
        self.output_filename_var.set("")
        self.output_filename_entry = Entry(self.samples_label_frame, textvariable=self.output_filename_var)
        self.output_filename_entry.grid(row=1, column=1)

        self.q_label = Label(self.samples_label_frame, text="Розмір вибірки:")
        self.q_label.grid(row=2, column=0, sticky="E", padx=5, pady=2)

        self.q_var = StringVar()
        self.q_var.set("")
        self.q_entry = Entry(self.samples_label_frame, textvariable=self.q_var, state=DISABLED)
        self.q_entry.grid(row=2, column=1)

        # 'Розмірності векторів'
        self.dimensions_label_frame.columnconfigure(1, weight=1)
        self.x1_label = Label(self.dimensions_label_frame, text="Розмірність X1:")
        self.x1_label.grid(row=0, column=0, sticky="E", padx=5, pady=2)
        self.x1_var = StringVar()
        self.x1_var.set("")
        self.x1_entry = Entry(self.dimensions_label_frame, textvariable=self.x1_var, state=DISABLED)
        self.x1_entry.grid(row=0, column=1, sticky="WE", padx=5, pady=2)

        self.x2_label = Label(self.dimensions_label_frame, text="Розмірність X2:")
        self.x2_label.grid(row=1, column=0, sticky="E", padx=5, pady=2)
        self.x2_var = StringVar()
        self.x2_var.set("")
        self.x2_spinbox = Entry(self.dimensions_label_frame, textvariable=self.x2_var, state=DISABLED)
        self.x2_spinbox.grid(row=1, column=1, sticky="WE", padx=5, pady=2)

        self.x3_label = Label(self.dimensions_label_frame, text="Розмірність X3:")
        self.x3_label.grid(row=2, column=0, sticky="E", padx=5, pady=2)
        self.x3_var = StringVar()
        self.x3_var.set("")
        self.x3_entry = Entry(self.dimensions_label_frame, textvariable=self.x3_var, state=DISABLED)
        self.x3_entry.grid(row=2, column=1, sticky="WE", padx=5, pady=2)

        self.y_label = Label(self.dimensions_label_frame, text="Розмірність Y:")
        self.y_label.grid(row=3, column=0, sticky="E", padx=5, pady=2)
        self.y_var = StringVar()
        self.y_var.set("")
        self.y_entry = Entry(self.dimensions_label_frame, textvariable=self.y_var, state=DISABLED)
        self.y_entry.grid(row=3, column=1, sticky="WE", padx=5, pady=2)

        # 'Поліноми'
        # 'Вид поліномів'
        self.polynomial_var = StringVar()
        self.polynomial_var.set(Polynom.CHEBYSHEV.name)
        self.chebyshev_radiobutton = Radiobutton(
            self.polynomials_type_label_frame,
            text="Поліноми Чебишева",
            variable=self.polynomial_var,
            value=Polynom.CHEBYSHEV.name,
        )
        self.chebyshev_radiobutton.grid(row=0, sticky="W")
        self.legandre_radiobutton = Radiobutton(
            self.polynomials_type_label_frame,
            text="Поліноми Лежандра",
            variable=self.polynomial_var,
            value=Polynom.LEGANDRE.name,
        )
        self.legandre_radiobutton.grid(row=1, sticky="W")
        self.lagerr_radiobutton = Radiobutton(
            self.polynomials_type_label_frame,
            text="Поліноми Лагерра",
            variable=self.polynomial_var,
            value=Polynom.LAGERR.name,
        )
        self.lagerr_radiobutton.grid(row=2, sticky="W")

        # 'Степені поліномів'
        self.polynomials_dimensions_label_frame.columnconfigure(1, weight=1)
        self.p1_label = Label(self.polynomials_dimensions_label_frame, text="P1:")
        self.p1_label.grid(row=0, column=0, sticky="E")
        self.p1_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p1_spinbox.grid(row=0, column=1, sticky="WE", padx=5, pady=2)

        self.p2_label = Label(self.polynomials_dimensions_label_frame, text="P2:")
        self.p2_label.grid(row=1, column=0, sticky="E")
        self.p2_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p2_spinbox.grid(row=1, column=1, sticky="WE", padx=5, pady=2)

        self.p3_label = Label(self.polynomials_dimensions_label_frame, text="P3:")
        self.p3_label.grid(row=2, column=0, sticky="E")
        self.p3_spinbox = Spinbox(self.polynomials_dimensions_label_frame, from_=1, to=4, width=5)
        self.p3_spinbox.grid(row=2, column=1, sticky="WE", padx=5, pady=2)

        # 'Додатково'
        # 'Ваги цільових функцій'
        self.weight = StringVar()
        self.weight.set(Weight.NORMED.name)
        self.normed_radiobutton = Radiobutton(
            self.weight_label_frame,
            text="Нормовані Yi",
            variable=self.weight,
            value=Weight.NORMED.name,
        )
        self.normed_radiobutton.grid(row=0, sticky="W")
        self.min_max_radiobutton = Radiobutton(
            self.weight_label_frame,
            text="(min(Yi) + max(Yi)) / 2",
            variable=self.weight,
            value=Weight.MIN_MAX.name,
        )
        self.min_max_radiobutton.grid(row=1, sticky="W")

        # 'Метод визначення лямбд'
        self.lambdas = StringVar()
        self.lambdas.set(Lambda.SINGLE_SET.name)
        self.single_set_radiobutton = Radiobutton(
            self.lambdas_label_frame,
            text="Одна система",
            variable=self.lambdas,
            value=Lambda.SINGLE_SET.name,
        )
        self.single_set_radiobutton.grid(row=0, sticky="W")
        self.triple_set_radiobutton = Radiobutton(
            self.lambdas_label_frame,
            text="Три системи",
            variable=self.lambdas,
            value=Lambda.TRIPLE_SET.name,
        )
        self.triple_set_radiobutton.grid(row=1, sticky="W")

        # 'Результати'
        self.result_area = ScrolledText(self.results_label_frame, height=44)
        self.result_area.grid(row=0, column=0, sticky="WENS")

        self.calculate_button = Button(
            self.additional_label_frame,
            text="Обрахувати",
            command=self.calculate,
            bg="#A6CF98",
            fg="white",
        )
        self.calculate_button.grid(sticky="WE", padx=5, pady=5, ipadx=5, ipady=5)

    def browse_file(self):
        samples_filename = filedialog.askopenfilename(
            title="Open a File",
            filetypes=(("Excel files", ".*xlsx"), ("All Files", "*.")),
        )
        self.samples_filename_var.set(samples_filename)
        return samples_filename

    def get_entries(self):
        samples_filename = None
        output_filename = "default"

        if self.samples_filename_entry.get() != "":
            samples_filename = self.samples_filename_entry.get()

        if self.output_filename_entry.get() != "":
            output_filename = self.output_filename_entry.get()

        P_dims = np.array(
            [
                int(self.p1_spinbox.get()),
                int(self.p2_spinbox.get()),
                int(self.p3_spinbox.get()),
            ]
        )
        return (
            samples_filename,
            output_filename,
            P_dims,
            self.polynomial_var.get(),
            self.weight.get(),
            self.lambdas.get(),
        )

    def calculate(self):
        s_f, o_f, P_dims, polynomial, weight, lambdas = self.get_entries()

        model = Model(s_f, o_f, P_dims, polynomial, weight, lambdas)

        self.q_var.set(str(model.Q))
        self.x1_var.set(str(model.X1_dim))
        self.x2_var.set(str(model.X2_dim))
        self.x3_var.set(str(model.X3_dim))
        self.y_var.set(str(model.Y_dim))

        model.calculate()

        self.result_area.insert(END, model.LOG)

        fig1, ax1 = plt.subplots(2, 3, figsize=(20, 10))
        fig1.suptitle("Порівняння значень вибірки і апроксимованих")

        fig2, ax2 = plt.subplots(2, 3, figsize=(20, 10))
        fig2.suptitle("Порівняння значень оригінальної вибірки і відновлених апроксимованих")
        for i in range(model.Y_dim):
            ax1[i // 3, i % 3].plot(model.Y_normed_array[:, i], label=f"Y_{i + 1}")
            ax1[i // 3, i % 3].plot(model.results_normed[:, i], label=f"F_{i + 1}")
            ax1[i // 3, i % 3].set_title(f"Ф{i + 1}")
            ax1[i // 3, i % 3].legend(loc=2)
            ax1[i // 3, i % 3].grid(True)

            ax2[i // 3, i % 3].plot(model.Y_array[:, i], label=f"Y_{i + 1}")
            ax2[i // 3, i % 3].plot(model.results[:, i], label=f"F_{i + 1}")
            ax2[i // 3, i % 3].set_title(f"Ф{i + 1}")
            ax2[i // 3, i % 3].legend(loc=2)
            ax2[i // 3, i % 3].grid(True)

        graphics2 = Toplevel(window)

        canvas = FigureCanvasTkAgg(fig2, master=graphics2)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0)

        graphics1 = Toplevel(window)

        canvas = FigureCanvasTkAgg(fig1, master=graphics1)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0)

        errors_label_frame = LabelFrame(graphics1, text="Нев'язка")
        errors_label_frame.columnconfigure(1, weight=1)
        errors_label_frame.grid(row=0, column=1)

        F1 = Label(errors_label_frame, text="Ф1:")
        F1.grid(row=0, column=0, sticky="E")
        f1 = StringVar()
        f1.set(str(model.residuals[0]))
        F1_spinbox = Entry(errors_label_frame, textvariable=f1)
        F1_spinbox.grid(row=0, column=1, sticky="WE", padx=5, pady=2)

        F2 = Label(errors_label_frame, text="Ф2:")
        F2.grid(row=1, column=0, sticky="E")
        f2 = StringVar()
        f2.set(str(model.residuals[1]))
        F2_spinbox = Entry(errors_label_frame, textvariable=f2)
        F2_spinbox.grid(row=1, column=1, sticky="WE", padx=5, pady=2)

        F3 = Label(errors_label_frame, text="Ф3:")
        F3.grid(row=2, column=0, sticky="E")
        f3 = StringVar()
        f3.set(str(model.residuals[2]))
        F3_spinbox = Entry(errors_label_frame, textvariable=f3)
        F3_spinbox.grid(row=2, column=1, sticky="WE", padx=5, pady=2)

        F4 = Label(errors_label_frame, text="Ф4:")
        F4.grid(row=3, column=0, sticky="E")
        f4 = StringVar()
        f4.set(str(model.residuals[3]))
        F4_spinbox = Entry(errors_label_frame, textvariable=f4)
        F4_spinbox.grid(row=3, column=1, sticky="WE", padx=5, pady=2)

        F4 = Label(errors_label_frame, text="Ф5:")
        F4.grid(row=4, column=0, sticky="E")
        f4 = StringVar()
        f4.set(str(model.residuals[4]))
        F4_spinbox = Entry(errors_label_frame, textvariable=f4)
        F4_spinbox.grid(row=4, column=1, sticky="WE", padx=5, pady=2)


application = Application(window)

window.mainloop()
