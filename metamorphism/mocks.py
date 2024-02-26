"""
experiment using mock to inject new behavior into existing
functions

"""
#%% target script
class Bookbinder:
    def bind_book(self, book):
        pass
        
def bind_book():
    b = Bookbinder()
    for book in ["The Hobbit", "The Silmarillion", "The Lord of the Rings"]:
        b.bind_book(book)

#%%
# now we want to write a decorator that will retrieve the list of books
# that has been bound without modifying the original function

from unittest.mock import patch

# Decorator to capture books
books_captured = []
def capture_books(func):
    """
    this decorator-like function is useful so that wrapper will no longer be calling
    the original function directly, which will cause infinite recursion
    """
    def wrapper(self, book, *args, **kwargs):
        books_captured.append(book)
        return func(self, book, *args, **kwargs)  # Call the original function
    return wrapper

from unittest.mock import patch

# normal call will not capture anything
bind_book()
print("nothing captured:", books_captured)

# with patch.object(Bookbinder, 'bind_book', new=capture_books(Bookbinder.bind_book)): # alternative
with patch('__main__.Bookbinder.bind_book', new=capture_books(Bookbinder.bind_book)):
    bind_book()
# this will contain captured book
print("captured:", books_captured)

# calling again will not make any difference outside the patch context
books_captured.clear()
bind_book()
print("nothing captured:", books_captured)

# This probably looks unremarkable, but the idea is quite new to me:
# now we are able to probe internal dynamics of function without modifying it

#%% alternative
books_captured.clear()

# normal call will not capture anything
bind_book()
print("nothing captured:", books_captured)

patcher = patch('__main__.Bookbinder.bind_book', new=capture_books(Bookbinder.bind_book))

patcher.start()
bind_book()
# this will contain captured book
print("captured:", books_captured)
patcher.stop()

# calling again will not make any difference outside the patch context
books_captured.clear()
bind_book()
print("nothing captured:", books_captured)

books_captured.clear()
from contextlib import ExitStack
with ExitStack() as stack:
    files = [stack.enter_context(patcher)]

    bind_book()
print("captured:", books_captured)