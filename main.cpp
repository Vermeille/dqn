#include <cstdio>
#include <utility>
#include <unistd.h>

template <class T>
class Point
{
    public:
        Point(T x, T y) : data_(x, y) {}
        T x() const { return data_.first; }
        void x(T x) { data_.first = x; }

        T y() const { return data_.second; }
        void y(T y) { data_.second = y; }
    private:
        std::pair<T, T> data_;
};

enum class Direction
{
    Up,
    Down,
    Left,
    Right,
    Stay
};

class Yop
{
    public:
        Yop(const Point<unsigned>& p) : pos_(p), move_(Direction::Up)
        {
        }

        const Point<unsigned>& GetPos() const { return pos_; }
        void SetPos(const Point<unsigned>& p) { pos_ = p; }
        Point<unsigned> NextPos() const
        {
            switch (move_)
            {
                case Direction::Stay:
                    return pos_;
                case Direction::Up:
                    return Point<unsigned>(pos_.x(), pos_.y() - 1);
                case Direction::Down:
                    return Point<unsigned>(pos_.x(), pos_.y() + 1);
                case Direction::Left:
                    return Point<unsigned>(pos_.x() - 1, pos_.y());
                case Direction::Right:
                    return Point<unsigned>(pos_.x() + 1, pos_.y());
            }
        }

    private:
        Point<unsigned> pos_;
        Direction move_;
};

class World
{
    public:
        World(unsigned w, unsigned h) : size_(w, h), yop_(Point<unsigned>(w / 2, h / 2)) {}

        void Draw()
        {
            printf("\x1B[0;0H");
            for (int j = 0; j < size_.x() + 2; ++j)
                putchar('-');
            putchar('\n');

            for (int i = 0; i < size_.y(); ++i)
            {
                putchar('|');
                for (int j = 0; j < size_.x(); ++j)
                {
                    if (yop_.GetPos().x() == j && yop_.GetPos().y() == i)
                        putchar('X');
                    else
                        putchar(' ');
                }
                putchar('|');
                putchar('\n');
            }

            for (int j = 0; j < size_.x() + 2; ++j)
                putchar('-');
            putchar('\n');
        }

        void Update()
        {
            auto move = yop_.NextPos();
            if (move.x() >= 0 && move.x() < size_.x() && move.y() >= 0 && move.y() < size_.y())
                yop_.SetPos(move);
        }

    private:
        const Point<unsigned int> size_;
        Yop yop_;
};

int main()
{
    World w(20, 20);
    while (1)
    {
        w.Update();
        usleep(500000);
        w.Draw();
    }
    return 0;
}
