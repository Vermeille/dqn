#include <OpenANN/OpenANN>
#include <OpenANN/io/DirectStorageDataSet.h>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "gnuplot-iostream.h"

static const int kWorldSize = 20;
static const int kMemorySize = 1000000;
static const int kReplaySize = 500;
static const int kRandMoveProbability = 100;
static const int kSaveFreq = 10000;

enum Direction
{
    UP, DOWN, LEFT, RIGHT
};

static Eigen::Matrix2i kUp;
static Eigen::Matrix2i kDown;
static Eigen::Matrix2i kLeft;
static Eigen::Matrix2i kRight;

void Init()
{
    kUp.row(0) << 0, -1;
    kDown.row(0) << 0, 1;
    kLeft.row(0) << -1, 0;
    kRight.row(0) << 1, 0;
    srand(time(nullptr));
}

Eigen::Matrix2i DirToMove(Direction d)
{
    switch (d)
    {
        case UP: return kUp;
        case DOWN: return kDown;
        case LEFT: return kLeft;
        case RIGHT: return kRight;
    }
}

struct State
{
    Eigen::Matrix2i food_pos;
    Eigen::Matrix2i yop_pos;

    State()
    {
        MoveFood();
        yop_pos.row(0) << rand() % kWorldSize, rand() % kWorldSize;
    }

    bool IsFoodEaten() const
    {
        return food_pos(0, 0) == yop_pos(0, 0) && food_pos(0, 1) == yop_pos(0, 1);
    }

    bool MoveIfPossible(Direction d)
    {
        Eigen::Matrix2i nxt = yop_pos + DirToMove(d);
        if (nxt(0, 0) >= 0 && nxt(0, 0) < kWorldSize && nxt(0, 1) >= 0 && nxt(0, 1) < kWorldSize)
        {
            yop_pos = nxt;
            return true;
        }
        else
        {
            return false;
        }
    }

    void MoveFood()
    {
        food_pos.row(0) << rand() % kWorldSize, rand() % kWorldSize;
    }

    int DistanceToFood()
    {
        return std::abs(food_pos(0, 0) - yop_pos(0, 0)) + std::abs(food_pos(0, 1) - yop_pos(0, 1));
    }

    Eigen::MatrixXd GetStateMatrix() const
    {
        Eigen::MatrixXd input(1, 4);
        input << food_pos(0, 0), food_pos(0, 1), yop_pos(0, 0), yop_pos(0, 1);
        return input;
    }
};

class Brain
{
    public:
        struct MemCell
        {
            int r;
            State state;
            int action;
            State next_state;

            MemCell(int other_r, const State& other_state, int other_action, const State& other_next_state)
                : r(other_r), state(other_state), action(other_action), next_state(other_next_state)
            { }
        };

        Eigen::MatrixXd ComputeTarget(const MemCell& mem)
        {
            Eigen::MatrixXd out = PredictRewards(mem.state);

            out(0, mem.action) = mem.r + 0.75 * PredictRewards(mem.next_state).maxCoeff();
            return out;
        }

        Brain() : score_(0), is_learning_(true)
        {
            try
            {
                brain_.load("brain.net");
            }
            catch (std::exception& e)
            {
                brain_.inputLayer(4)
                    .fullyConnectedLayer(10, OpenANN::RECTIFIER)
                    .outputLayer(4, OpenANN::LINEAR);
            }
        }

        void Save() { brain_.save("brain.net"); }

        Eigen::MatrixXd PredictRewards(const State& s)
        {
            return brain_(s.GetStateMatrix());
        }

        Direction ComputeMove(const State& s)
        {
            Eigen::MatrixXd out = PredictRewards(s);

            double max = out(0, 0);
            int idx = 0;
            for (int i = 1; i < 4; ++i)
                if (out(0, i) > max)
                {
                    max = out(0, i);
                    idx = i;
                }

            if (rand() % kRandMoveProbability == 0)
                idx = rand() % 4;

            return static_cast<Direction>(idx);
        }

        int GetScore() const { return score_; }

        void Reward(int r, const State& state, int action, const State& next_state)
        {
            score_ += r;

            if (!is_learning_)
                return;

            if (memory_.size() == kMemorySize)
                memory_[rand() % kMemorySize] = MemCell{r, state, action, next_state};
            else
                memory_.emplace_back(r, state, action, next_state);

            if (memory_.size() < kReplaySize)
                return;

            Eigen::MatrixXd inputs(kReplaySize, 4);
            Eigen::MatrixXd outputs(kReplaySize, 4);

            for (int i = 0; i < kReplaySize; ++i)
            {
                const MemCell& mem = memory_[rand() % memory_.size()];
                inputs.row(i) = mem.state.GetStateMatrix();
                outputs.row(i) = ComputeTarget(mem);
            }

            OpenANN::DirectStorageDataSet dataset(&inputs, &outputs);
            brain_.trainingSet(dataset);
            OpenANN::StoppingCriteria stop;
            stop.maximalIterations = 5;
            OpenANN::train(brain_, "LMA", OpenANN::MSE, stop);
        }

        void StopLearning() { is_learning_ = false; }

    std::vector<MemCell> memory_;
    int score_;
    bool is_learning_;
    OpenANN::Net brain_;
};

void Draw(const State& world)
{
    fputs("\x1B[2J\x1B[0;0H", stdout);
    for (int j = 0; j < kWorldSize + 2; ++j)
        fputc('-', stdout);
    fputc('\n', stdout);

    for (int i = 0; i < kWorldSize; ++i)
    {
        fputc('|', stdout);
        for (int j = 0; j < kWorldSize; ++j)
        {
            if (j == world.food_pos(0, 0) && i == world.food_pos(0, 1))
                fputc('O', stdout);
            else if (j == world.yop_pos(0, 0) && i == world.yop_pos(0, 1))
                fputc('X', stdout);
            else
                fputc(' ', stdout);
        }
        fputs("|\n", stdout);
    }

    for (int j = 0; j < kWorldSize + 2; ++j)
        fputc('-', stdout);
    fputc('\n', stdout);
}

bool HasOption(const char* str, int argc, char** argv)
{
    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(str, argv[i]))
            return true;
    }
    return false;
}

int main(int argc, char** argv)
{
    Init();

    State state;
    Brain yop;
    std::ofstream history("score.dat");

    unsigned int iter = 0;
    int last_score = 0;

    bool draw = !HasOption("--no-draw", argc, argv);
    bool demo = HasOption("--demo", argc, argv);
    if (demo || HasOption("--no-learning", argc, argv))
        yop.StopLearning();

    Gnuplot g;
    while (true)
    {
        if (draw)
        {
            Draw(state);
            std::cout << "iter: " << iter << "\n";
            std::cout << "score: " << yop.GetScore() << "\n";
        }
        else
        {
            if (iter % 1000 == 0)
            {
                history << yop.GetScore() - last_score << "\n";
                history.flush();
                g << "plot \"score.dat\" with lines\n";
                std::cout << (iter / 1000) << ", " << yop.GetScore() - last_score << "\n";
                last_score = yop.GetScore();
            }
        }

        Direction move = yop.ComputeMove(state);
        State next_state = state;

        if (!next_state.MoveIfPossible(move))
        {
            yop.Reward(-1, state, move, next_state);
        }
        else if (next_state.IsFoodEaten())
        {
            next_state.MoveFood();
            yop.Reward(1, state, move, next_state);
        }
        else
        {
            int win = state.DistanceToFood() - next_state.DistanceToFood();
            win = std::max(-1, std::min(win, 1));
            yop.Reward(win, state, move, next_state);
        }

        state = next_state;

        if (demo)
            usleep(100000UL);
        if (iter % kSaveFreq == 0)
            yop.Save();

        ++iter;
    }

    return 0;
}
