import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QCheckBox, QLabel, QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QComboBox, QLineEdit, QSpinBox, QStackedWidget, QStatusBar,
    QDoubleSpinBox, QAbstractItemView
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import solver

MAX_POINTS = 20
FUNCTIONS = ["sin(x)", "cos(x)", "exp(x)"]


def set_modern_light_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f0f2f5"))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f7f9fc"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.Button, QColor("#e1e5eb"))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#0078d7"))
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
    app.setPalette(palette)

    app.setStyleSheet("""
        QWidget {
            font-family: Segoe UI, sans-serif;
            font-size: 13px;
            color: #000000;
        }
        QMainWindow {
            background-color: #f0f2f5;
        }
        QPushButton {
            background-color: #0078d7;
            color: white;
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #005fa1;
        }
        QLineEdit, QComboBox {
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 4px;
        }
        QTableWidget {
            background-color: white;
            border: 1px solid #ccc;
        }
        QHeaderView::section {
            background-color: #e1e5eb;
            padding: 4px;
            border: 1px solid #d0d0d0;
        }
    """)


class InterpolatorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Интерполятор")
        self.resize(1200, 700)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        root.addLayout(top, stretch=8)

        left_panel = QVBoxLayout()
        top.addLayout(left_panel, stretch=2)

        src_box = QGroupBox("Ввод данных")
        sl = QVBoxLayout(src_box)
        self.rb_table = QRadioButton("Таблица")
        self.rb_file = QRadioButton("Файл")
        self.rb_func = QRadioButton("Функция")
        self.rb_table.setChecked(True)
        for rb in (self.rb_table, self.rb_file, self.rb_func):
            sl.addWidget(rb)
            rb.toggled.connect(self._switch_page)
        left_panel.addWidget(src_box)

        meth_box = QGroupBox("Методы интерполяции")
        ml = QVBoxLayout(meth_box)
        self.cb_lagr = QCheckBox("Лагранж")
        self.cb_newton = QCheckBox("Ньютон")
        self.cb_gauss = QCheckBox("Гаусс")
        self.cb_stirling = QCheckBox("Стирлинг")
        self.cb_bessel = QCheckBox("Бессель")
        for cb in (self.cb_lagr, self.cb_newton, self.cb_gauss,
                   self.cb_stirling, self.cb_bessel):
            ml.addWidget(cb)
        btn_all = QPushButton("Выбрать всё");
        btn_all.clicked.connect(self._select_all)
        ml.addWidget(btn_all)
        left_panel.addWidget(meth_box)

        self.figure = Figure(figsize=(5, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        top.addWidget(self.canvas, stretch=5)

        mid = QHBoxLayout();
        root.addLayout(mid, stretch=5)

        inp_layout = QVBoxLayout()
        mid.addLayout(inp_layout, stretch=3)

        self.pages = QStackedWidget()
        inp_layout.addWidget(self.pages)
        self._page_table()
        self._page_file()
        self._page_func()

        xrow = QHBoxLayout()
        xrow.addWidget(QLabel("x* ="))
        self.sb_xstar = QDoubleSpinBox()
        self.sb_xstar.setRange(-1e6, 1e6)
        self.sb_xstar.setDecimals(6)
        self.sb_xstar.setValue(0.0)
        xrow.addWidget(self.sb_xstar)
        inp_layout.addLayout(xrow)

        tbl_layout = QVBoxLayout()
        mid.addLayout(tbl_layout, stretch=2)

        diff_box = QGroupBox("Таблица конечных разностей")
        dl = QVBoxLayout(diff_box)
        self.tbl_diffs = QTableWidget()
        self.tbl_diffs.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        dl.addWidget(self.tbl_diffs)
        tbl_layout.addWidget(diff_box)

        res_box = QGroupBox("Результаты")
        rl = QVBoxLayout(res_box)
        self.tbl_results = QTableWidget(0, 2)
        self.tbl_results.setHorizontalHeaderLabels(["Метод", "Значение"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        rl.addWidget(self.tbl_results)
        tbl_layout.addWidget(res_box)

        bottom = QHBoxLayout();
        root.addLayout(bottom, stretch=1)
        self.status = QStatusBar();
        bottom.addWidget(self.status, stretch=5)
        btn_solve = QPushButton("Решить");
        btn_solve.clicked.connect(self._solve)
        bottom.addWidget(btn_solve, stretch=1)

    def _page_table(self):
        w = QWidget();
        l = QVBoxLayout(w)
        self.tbl_input = QTableWidget(3, 2)
        self.tbl_input.setHorizontalHeaderLabels(["x", "y"])
        l.addWidget(self.tbl_input)
        btns = QHBoxLayout();
        btns.addStretch()
        plus = QPushButton("+ строка");
        plus.clicked.connect(self._add_row)
        minus = QPushButton("– строка");
        minus.clicked.connect(self._del_row)
        btns.addWidget(plus);
        btns.addWidget(minus);
        btns.addStretch()
        l.addLayout(btns)
        self.pages.addWidget(w)

    def _page_file(self):
        w = QWidget();
        l = QVBoxLayout(w)
        self.le_path = QLineEdit();
        self.le_path.setReadOnly(True)
        btn = QPushButton("Обзор…");
        btn.clicked.connect(self._browse)
        row = QHBoxLayout();
        row.addWidget(self.le_path);
        row.addWidget(btn)
        l.addLayout(row)
        self.pages.addWidget(w)

    def _page_func(self):
        w = QWidget();
        l = QVBoxLayout(w)
        row1 = QHBoxLayout()
        self.cmb_func = QComboBox();
        self.cmb_func.addItems(FUNCTIONS)
        row1.addWidget(QLabel("f(x) ="));
        row1.addWidget(self.cmb_func)
        l.addLayout(row1)
        row2 = QHBoxLayout()
        val = QDoubleValidator()
        self.le_left = QLineEdit("-3.14");
        self.le_left.setValidator(val)
        self.le_right = QLineEdit(" 3.14");
        self.le_right.setValidator(val)
        row2.addWidget(QLabel("От"));
        row2.addWidget(self.le_left)
        row2.addWidget(QLabel("До"));
        row2.addWidget(self.le_right)
        l.addLayout(row2)
        row3 = QHBoxLayout()
        self.sb_n = QSpinBox();
        self.sb_n.setRange(2, MAX_POINTS);
        self.sb_n.setValue(5)
        row3.addWidget(QLabel("N точек"));
        row3.addWidget(self.sb_n)
        l.addLayout(row3)
        self.pages.addWidget(w)

    def _select_all(self):
        for cb in (self.cb_lagr, self.cb_newton, self.cb_gauss,
                   self.cb_stirling, self.cb_bessel):
            cb.setChecked(True)

    def _switch_page(self):
        if self.rb_table.isChecked():
            self.pages.setCurrentIndex(0)
        elif self.rb_file.isChecked():
            self.pages.setCurrentIndex(1)
        else:
            self.pages.setCurrentIndex(2)

    def _add_row(self):
        if self.tbl_input.rowCount() < MAX_POINTS:
            self.tbl_input.insertRow(self.tbl_input.rowCount())
        else:
            self.status.showMessage(f"Максимум {MAX_POINTS} точек", 5000)

    def _del_row(self):
        rows = {idx.row() for idx in self.tbl_input.selectedIndexes()}
        if not rows and self.tbl_input.rowCount() > 1:
            self.tbl_input.removeRow(self.tbl_input.rowCount() - 1)
        elif rows:
            self.tbl_input.removeRow(max(rows))

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть файл", "", "Text files (*.txt *.csv)")
        if path:
            self.le_path.setText(path)

    def _solve(self):
        self.status.clearMessage()
        try:
            if self.rb_table.isChecked():
                pts = []
                for r in range(self.tbl_input.rowCount()):
                    xi = self.tbl_input.item(r, 0)
                    yi = self.tbl_input.item(r, 1)
                    if xi is None or yi is None:
                        raise ValueError("Заполните все ячейки")
                    pts.append((float(xi.text()), float(yi.text())))
                data_kind = 'table'
                data = pts
            elif self.rb_file.isChecked():
                if not self.le_path.text():
                    raise ValueError("Файл не выбран")
                data_kind = 'file'
                data = self.le_path.text()
            else:
                left = float(self.le_left.text())
                right = float(self.le_right.text())
                if right <= left:
                    raise ValueError("Правая граница ≤ левой")
                data_kind = 'func'
                data = {
                    'name': self.cmb_func.currentText(),
                    'left': left,
                    'right': right,
                    'n': self.sb_n.value()
                }
        except Exception as e:
            self.status.showMessage(f"Ошибка: {e}", 5000)
            return

        methods = {
            'lagrange': self.cb_lagr.isChecked(),
            'newton': self.cb_newton.isChecked(),
            'gauss': self.cb_gauss.isChecked(),
            'stirling': self.cb_stirling.isChecked(),
            'bessel': self.cb_bessel.isChecked()
        }

        solver.process_data(data_kind, data, methods, self.sb_xstar.value(), self)

    def clear_diff_table(self):
        self.tbl_diffs.clearContents()
        self.tbl_diffs.setRowCount(0)
        self.tbl_diffs.setColumnCount(0)

    def update_diff_table(self, diffs):
        cols = len(diffs)
        rows = len(diffs[0]) if cols else 0
        self.tbl_diffs.setRowCount(rows)
        self.tbl_diffs.setColumnCount(cols)
        headers = ["y"] + [f"Δ^{i}" for i in range(1, cols)]
        self.tbl_diffs.setHorizontalHeaderLabels(headers)
        for c, col in enumerate(diffs):
            for r, val in enumerate(col):
                it = QTableWidgetItem(f"{val:.6g}")
                it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tbl_diffs.setItem(r, c, it)
        self.tbl_diffs.resizeColumnsToContents()
        self.tbl_diffs.resizeRowsToContents()

    def clear_results(self):
        self.tbl_results.setRowCount(0)

    def add_result(self, method, value):
        r = self.tbl_results.rowCount()
        self.tbl_results.insertRow(r)
        self.tbl_results.setItem(r, 0, QTableWidgetItem(method))
        self.tbl_results.setItem(r, 1, QTableWidgetItem(value))

    def show_error(self, msg):
        self.status.showMessage(f"Ошибка: {msg}", 10000)

    def show_ok(self, msg):
        self.status.showMessage(msg, 5000)

    def plot(self, points, x0):
        xs, ys = zip(*points)
        self.ax.clear()
        self.ax.scatter(xs, ys, label="Узлы")

        xx = [xs[0] + i * (xs[-1] - xs[0]) / 300 for i in range(301)]
        yy_n = [solver.interp_newton(points, x) for x in xx]
        self.ax.plot(xx, yy_n, linestyle="--", label="Ньютон")

        yy_g = [solver.interp_gauss(points, x) for x in xx]
        self.ax.plot(xx, yy_g, linestyle="-.", label="Гаусс")

        yy_s = [solver.interp_stirling(points, x) for x in xx]
        self.ax.plot(xx, yy_s, linestyle=":", label="Стирлинг")

        yy_b = [solver.interp_bessel(points, x) for x in xx]
        self.ax.plot(xx, yy_b, linestyle="--", label="Бессель")

        yy_l = [solver.interp_lagrange(points, x) for x in xx]
        self.ax.plot(xx, yy_l, linestyle="--", label="Лагранж")

        y0 = solver.interp_newton(points, x0)
        self.ax.scatter([x0], [y0], marker="x", s=100, label=f"x*={x0:.4g}")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Интерполяция")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    set_modern_light_theme(app)
    gui = InterpolatorGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
