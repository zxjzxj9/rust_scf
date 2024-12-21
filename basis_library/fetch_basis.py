#! /usr/bin/env python

import requests
import json
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import curses
from curses import wrapper
import logging

# Match Rust enums and structs
class ShellType(Enum):
    S = 0  # l = 0
    P = 1  # l = 1
    D = 2  # l = 2
    F = 3  # l = 3

@dataclass
class GTO:
    exponent: float
    coefficients: List[float]  # Changed to handle multiple coefficients
    angular_momentum: List[int]  # Added to store l values

@dataclass
class BasisSet:
    shells: List[GTO]
    element: int  # Added to store atomic number

def parse_basis_set(json_data: Dict, element_number: str) -> BasisSet:
    shells = []

    # Get the electron shells for the specific element
    element_data = json_data['elements'][element_number]['electron_shells']

    for shell in element_data:
        # Convert exponents to floats
        exponents = [float(exp) for exp in shell['exponents']]

        # Get angular momentum values
        angular_momentum = shell['angular_momentum']

        # Get coefficients for all angular momentum components
        coefficient_sets = shell['coefficients']

        # For each exponent, create a GTO with all its coefficients
        for i in range(len(exponents)):
            coeffs = [float(coeff_set[i]) for coeff_set in coefficient_sets]
            gto = GTO(
                exponent=exponents[i],
                coefficients=coeffs,
                angular_momentum=angular_momentum
            )
            shells.append(gto)

    return BasisSet(
        shells=shells,
        element=int(element_number)
    )

logging.basicConfig(filename='simple_curses_debug.log', level=logging.DEBUG)

class SelectionMenu:
    def __init__(self, items, title="Selection Menu", subtitle=None):
        self.items = list(items)
        self.title = title
        self.current_position = 0
        self.window = None
        self.page_offset = 0
        self.items_per_page = 0

        if subtitle is None:
            self.subtitle = "Page [{current_page}/{total_pages}] | ↑↓/j/k: Navigate | PgUp/PgDn: Change Page | Enter: Select | q: Quit"
        else:
            self.subtitle = subtitle

    def setup_window(self, stdscr):
        self.window = stdscr
        curses.start_color()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Selected item
        curses.curs_set(0)  # Hide cursor
        height, _ = self.window.getmaxyx()
        self.items_per_page = height - 4  # Reserve lines for title, subtitle, and status

    def get_current_page_items(self):
        start_idx = self.page_offset
        end_idx = min(start_idx + self.items_per_page, len(self.items))
        return list(self.items)[start_idx:end_idx]

    def get_page_info(self):
        total_pages = (len(self.items) + self.items_per_page - 1) // self.items_per_page
        current_page = (self.page_offset // self.items_per_page) + 1
        return current_page, total_pages

    def draw(self):
        try:
            self.window.clear()
            height, width = self.window.getmaxyx()

            # Draw title
            self.window.addnstr(0, 0, self.title, width-1)

            # Draw subtitle with page information
            current_page, total_pages = self.get_page_info()
            subtitle = self.subtitle.format(current_page=current_page, total_pages=total_pages)
            self.window.addnstr(1, 0, subtitle, width-1)

            # Draw items for current page
            visible_items = self.get_current_page_items()
            for idx, item in enumerate(visible_items):
                y = idx + 2  # Start after header
                if y >= height - 1:  # Leave one line for status
                    break

                # Get relative and absolute position
                abs_pos = idx + self.page_offset
                item_str = str(item)[:width-1]  # Truncate if too long

                try:
                    if abs_pos == self.current_position:
                        self.window.attron(curses.color_pair(1))
                        self.window.addnstr(y, 0, f" {item_str}", width-1)
                        self.window.attroff(curses.color_pair(1))
                    else:
                        self.window.addnstr(y, 0, f" {item_str}", width-1)
                except curses.error:
                    pass

            self.window.refresh()
        except curses.error:
            pass

    def handle_input(self, key):
        current_page, total_pages = self.get_page_info()
        items_on_current_page = len(self.get_current_page_items())

        if key in [curses.KEY_UP, ord('k')]:
            if self.current_position > 0:
                self.current_position -= 1
                # If moved above current page, go to previous page
                if self.current_position < self.page_offset:
                    self.page_offset = max(0, self.page_offset - self.items_per_page)

        elif key in [curses.KEY_DOWN, ord('j')]:
            if self.current_position < len(self.items) - 1:
                self.current_position += 1
                # If moved below current page, go to next page
                if self.current_position >= self.page_offset + self.items_per_page:
                    self.page_offset = min(
                        len(self.items) - self.items_per_page,
                        self.page_offset + self.items_per_page
                    )

        elif key in [curses.KEY_PPAGE, ord('b')]:  # Page Up
            if current_page > 1:
                self.page_offset = max(0, self.page_offset - self.items_per_page)
                self.current_position = self.page_offset

        elif key in [curses.KEY_NPAGE, ord('f')]:  # Page Down
            if current_page < total_pages:
                self.page_offset = min(
                    len(self.items) - self.items_per_page,
                    self.page_offset + self.items_per_page
                )
                self.current_position = self.page_offset

        elif key == ord('\n'):  # Enter
            return self.items[self.current_position]
        elif key == ord('q'):
            return None

        # Ensure current_position stays within current page bounds
        if self.current_position < self.page_offset:
            self.current_position = self.page_offset
        elif self.current_position >= self.page_offset + items_on_current_page:
            self.current_position = self.page_offset + items_on_current_page - 1

        return False

    def run(self, stdscr):
        self.setup_window(stdscr)
        while True:
            self.draw()
            key = self.window.getch()
            result = self.handle_input(key)
            if result is not None:
                return result

def main(stdscr):
    # Basic setup
    curses.use_default_colors()

    # Fetch data
    base_url = "https://www.basissetexchange.org"
    try:
        r = requests.get(base_url + '/api/metadata')
        meta_data = r.json()
    except requests.RequestException as e:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Error fetching data: {str(e)}")
        stdscr.refresh()
        stdscr.getch()
        return

    # Create and run menu
    menu = SelectionMenu(
        items=meta_data.keys(),
        title="### Available Basis Sets ###"
    )

    selected_basis = menu.run(stdscr)

    # Show selection result
    if selected_basis:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Selected: {selected_basis}")
        stdscr.addstr(1, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()

if __name__ == '__main__':
    wrapper(main)